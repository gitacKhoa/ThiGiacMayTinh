"""Rendering/export adapters. No windows or input prompts; images are returned to callers."""

import csv
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from src.configuration import config_to_dict
from src.calibration.image_geometry import geometry_sidecar_path, save_undistorted_image
from src.edge_detection.subpixel_edge import draw_subpixel_edges
from src.measurement.homography import points_to_mm
from src.segmentation.segment import as_gray


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def save_result(output, image, result, metadata=None):
    """Compatibility export for the original point-cloud-only API."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    image_path = output / 'undistorted.png'
    geometry = (metadata or {}).get('preprocessing')
    if geometry is not None:
        save_undistorted_image(image_path, image, geometry)
    else:
        if not cv2.imwrite(str(image_path), image):
            raise OSError(f'Cannot write {image_path}')
        # A rewritten unverified image must not inherit a previous run's sidecar.
        geometry_sidecar_path(image_path).unlink(missing_ok=True)
    previews = {'mask.png': result.segmentation.mask,
                'edges.png': result.edges.edge_map, 'overlay.png': draw_subpixel_edges(image, result.edges.points)}
    for name, preview in previews.items():
        if not cv2.imwrite(str(output / name), preview):
            raise OSError(f'Cannot write {output / name}')
    world = result.points_mm if result.points_mm is not None else np.empty((0, 2), dtype=np.float64)
    H = result.homography if result.homography is not None else np.empty((0, 0), dtype=np.float64)
    np.savez_compressed(output / 'edges.npz', points_px=result.edges.points,
                        pixel_points=result.edges.pixel_points, normals=result.edges.normals,
                        offsets=result.edges.offsets, strengths=result.edges.strengths, points_mm=world, homography=H)
    columns = [result.edges.points, result.edges.offsets[:, None], result.edges.strengths[:, None]]
    header = 'x_px,y_px,normal_offset_px,gradient_strength'
    if result.points_mm is not None:
        columns.append(result.points_mm)
        header += ',x_mm,y_mm'
    np.savetxt(output / 'edges.csv', np.column_stack(columns), delimiter=',', header=header, comments='', fmt='%.10f')
    summary = dict(metadata or {})
    summary.update(coordinate_frame='undistorted.png; x=column, y=row; no ROI offset needed',
                   image_size=[image.shape[1], image.shape[0]], roi=list(result.segmentation.roi),
                   component_count=result.segmentation.component_count, threshold=result.segmentation.threshold,
                   candidate_count=result.edges.candidate_count, point_count=len(result.edges.points),
                   rejected_count=result.edges.candidate_count - len(result.edges.points),
                   has_mm=result.points_mm is not None, homography_rmse_mm=result.homography_rmse_mm,
                   point_order='unordered; do not use as a contour for area or perimeter')
    summary.setdefault('measurement_status', 'edge coordinates only; dimensions not requested')
    _write_json(output / 'metadata.json', summary)


def inspection_to_dict(result):
    """UI/JSON-friendly result without image arrays or hundreds of raw edge points."""
    measurement = result.measurement
    model = None
    if measurement is not None:
        model = asdict(measurement)
        model.pop('inliers')
        model['inlier_count'] = int(measurement.inliers.sum())
        model['input_point_count'] = len(measurement.inliers)
    uncertainties = {k: asdict(v) for k, v in result.uncertainties.items()}
    for uncertainty in uncertainties.values():
        if not uncertainty['available']:
            uncertainty.update(standard_mm=None, expanded_mm=None)
    return _jsonable(dict(version=1, units='mm', measurement=model,
                          uncertainties=uncertainties,
                          tolerances={k: asdict(v) for k, v in result.tolerance_results.items()},
                          qc_status=result.qc_status, warnings=result.warnings))


def render_inspection(image, result):
    """Draw fitted geometry in original pixel coordinates; add a separate text panel.

    The input image is not resized or mutated. GUI can alternatively use the
    model's outline/endpoints and inverse H to implement its own overlays.
    """
    gray = as_gray(image)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image[:, :, :3].copy()
    text = ['TOLERANCE VISION', f'QC: {result.qc_status}']
    if result.measurement is not None:
        m = result.measurement
        inverse = np.linalg.inv(result.processing.homography)
        def pixels(points):
            p = points_to_mm(np.asarray(points, dtype=float), inverse)
            # Protect drawing int32 from extreme coordinates; no change to numeric report.
            return np.rint(np.clip(p, -1000000, 1000000)).astype(np.int32)
        outline = pixels(m.outline_mm)
        cv2.polylines(out, [outline.reshape(-1, 1, 2)], True, (0, 220, 0), 2, cv2.LINE_AA)
        text += [f'Model: {m.kind}', (f'Fit RMSE: {m.fit_rmse_mm:.4f} mm'
                                    if m.fit_rmse_mm is not None else 'Overall extents (not fitted sides)')]
        for name, dimension in m.dimensions.items():
            a, b = pixels(dimension.endpoints_mm)
            cv2.line(out, tuple(a), tuple(b), (255, 100, 0), 2, cv2.LINE_AA)
            for p in (a, b):
                cv2.circle(out, tuple(p), 4, (255, 100, 0), -1)
            u = result.uncertainties[name]
            uncertainty_text = (f'U = {u.expanded_mm:.4f} mm (k={u.coverage_factor:g})'
                                if m.kind != 'bounding-box' or u.validated or u.expanded_mm > 0
                                else 'U: unknown (external budget needed)')
            text += ['', f'{name}: {dimension.value_mm:.3f} mm', uncertainty_text]
            if name in result.tolerance_results:
                t = result.tolerance_results[name]
                text += [f'Limits: {t.lower_limit_mm:g} .. {t.upper_limit_mm:g} mm',
                         f'Deviation: {t.deviation_mm:+.4f} mm', f'{name}: {t.status}']
        text += ['', 'Uncertainty: ' + ('user-validated budget' if all(u.validated for u in result.uncertainties.values())
                                       else 'PROVISIONAL / incomplete budget'),
                 'Board RMSE is NOT measurement accuracy.']
    else:
        text += ['', 'Edge-only mode: no dimensions requested.']
    height = max(out.shape[0], 32 + len(text) * 25)
    canvas = np.full((height, out.shape[1] + 490, 3), 245, np.uint8)
    canvas[:out.shape[0], :out.shape[1]] = out
    for i, line in enumerate(text):
        cv2.putText(canvas, line, (out.shape[1] + 15, 27 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (30, 30, 30), 1, cv2.LINE_AA)
    return canvas


def save_inspection(output, image, result, *, config=None, metadata=None):
    """Write current-run measurement outputs even in none mode (no stale mm/QC report)."""
    output = Path(output)
    report = inspection_to_dict(result)
    if config is not None:
        report['configuration'] = config_to_dict(config)
    report['source'] = dict(metadata or {})
    summary = dict(metadata or {})
    summary.update(measurement_status='dimensions computed' if result.measurement is not None else 'edge-only mode',
                   measurement_kind=result.measurement.kind if result.measurement is not None else 'none',
                   qc_status=result.qc_status)
    save_result(output, image, result.processing, summary)
    _write_json(output / 'measurements.json', report)
    with (output / 'measurements.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['dimension', 'value_mm', 'standard_uncertainty_mm', 'expanded_uncertainty_mm',
                         'coverage_factor', 'uncertainty_validated', 'nominal_mm', 'lower_limit_mm',
                         'upper_limit_mm', 'deviation_mm', 'status'])
        if result.measurement is not None:
            for name, dimension in result.measurement.dimensions.items():
                u = result.uncertainties[name]
                t = result.tolerance_results.get(name)
                writer.writerow([name, dimension.value_mm, u.standard_mm if u.available else '',
                                 u.expanded_mm if u.available else '', u.coverage_factor,
                                 u.validated, t.nominal_mm if t else '', t.lower_limit_mm if t else '',
                                 t.upper_limit_mm if t else '', t.deviation_mm if t else '',
                                 t.status if t else 'NOT_CONFIGURED'])
    if not cv2.imwrite(str(output / 'measurement_overlay.png'), render_inspection(image, result)):
        raise OSError('Cannot write measurement_overlay.png')
