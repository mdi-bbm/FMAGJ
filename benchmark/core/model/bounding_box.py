from benchmark.core.model.base_model import PydanticFrozen


class BoundingBox2D(PydanticFrozen):
    left: float
    top: float
    width: float
    height: float

    def area(self) -> float:
        return self.width * self.height

    def intersection_area(self, other: "BoundingBox2D") -> float:
        x1 = max(self.left, other.left)
        y1 = max(self.top, other.top)
        x2 = min(self.left + self.width, other.left + other.width)
        y2 = min(self.top + self.height, other.top + other.height)
        return max(0., x2 - x1) * max(0., y2 - y1)

    def union_area(self, other: "BoundingBox2D") -> float:
        return self.area() + other.area() - self.intersection_area(other)

    def iou(self, other: "BoundingBox2D") -> float:
        inter = self.intersection_area(other)
        union = self.union_area(other)
        return inter / union if union > 0. else 0.
