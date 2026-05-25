from __future__ import annotations

from app.models.schemas import OCRResult


class OCRPostprocessor:
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        line_height_tolerance: float = 0.5,
        paragraph_gap_factor: float = 1.8,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.line_height_tolerance = line_height_tolerance
        self.paragraph_gap_factor = paragraph_gap_factor

    def filter_by_confidence(self, results: list[OCRResult]) -> list[OCRResult]:
        return [r for r in results if r.confidence >= self.confidence_threshold]

    def _bbox_center_y(self, bbox: list[list[float]]) -> float:
        return sum(pt[1] for pt in bbox) / len(bbox)

    def _bbox_min_x(self, bbox: list[list[float]]) -> float:
        return min(pt[0] for pt in bbox)

    def _bbox_height(self, bbox: list[list[float]]) -> float:
        ys = [pt[1] for pt in bbox]
        return max(ys) - min(ys)

    def sort_reading_order(self, results: list[OCRResult]) -> list[OCRResult]:
        """Sort detections in top-to-bottom, left-to-right reading order."""
        if not results:
            return results

        avg_height = sum(self._bbox_height(r.bbox) for r in results) / len(results)
        tolerance = avg_height * self.line_height_tolerance

        sorted_results = sorted(results, key=lambda r: self._bbox_center_y(r.bbox))

        lines: list[list[OCRResult]] = []
        current_line: list[OCRResult] = [sorted_results[0]]
        current_y = self._bbox_center_y(sorted_results[0].bbox)

        for result in sorted_results[1:]:
            center_y = self._bbox_center_y(result.bbox)
            if abs(center_y - current_y) <= tolerance:
                current_line.append(result)
            else:
                lines.append(current_line)
                current_line = [result]
                current_y = center_y
        lines.append(current_line)

        ordered: list[OCRResult] = []
        for line in lines:
            line.sort(key=lambda r: self._bbox_min_x(r.bbox))
            ordered.extend(line)

        return ordered

    def group_paragraphs(self, results: list[OCRResult]) -> list[list[OCRResult]]:
        """Group sorted results into paragraphs based on vertical gaps."""
        if not results:
            return []

        avg_height = sum(self._bbox_height(r.bbox) for r in results) / len(results)
        paragraph_gap = avg_height * self.paragraph_gap_factor

        paragraphs: list[list[OCRResult]] = [[results[0]]]

        for prev, curr in zip(results, results[1:]):
            prev_bottom = max(pt[1] for pt in prev.bbox)
            curr_top = min(pt[1] for pt in curr.bbox)
            gap = curr_top - prev_bottom

            if gap > paragraph_gap:
                paragraphs.append([curr])
            else:
                paragraphs[-1].append(curr)

        return paragraphs

    def to_structured_text(self, results: list[OCRResult]) -> str:
        if not results:
            return ""

        ordered = self.sort_reading_order(results)
        paragraphs = self.group_paragraphs(ordered)

        paragraph_texts: list[str] = []
        for paragraph in paragraphs:
            line_text = " ".join(r.text for r in paragraph)
            paragraph_texts.append(line_text)

        return "\n\n".join(paragraph_texts)
