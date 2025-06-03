
from math import sqrt
from typing import List, Tuple, Optional, Union
from pathlib import Path

import norfair
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from soccer.utils import get_bbox_center, round_tuple_coords # Assuming utils.py is in the same dir
from config_loader import config

# It's better to load fonts once or make them configurable
DEFAULT_FONT_PATH = Path(config['paths']['font'])
try:
    DEFAULT_FONT_REGULAR_20 = PIL.ImageFont.truetype(str(DEFAULT_FONT_PATH), size=20)
    DEFAULT_FONT_REGULAR_24 = PIL.ImageFont.truetype(str(DEFAULT_FONT_PATH), size=24)
except IOError:
    print(f"Warning: Font {DEFAULT_FONT_PATH} not found. Using default PIL font.")
    DEFAULT_FONT_REGULAR_20 = PIL.ImageFont.load_default()
    DEFAULT_FONT_REGULAR_24 = PIL.ImageFont.load_default()


class Draw:
    @staticmethod
    def draw_rectangle(
        img: PIL.Image.Image,
        origin: Tuple[int, int],
        width: int,
        height: int,
        color: Tuple[int, int, int],
        thickness: int = 2,
    ) -> PIL.Image.Image:
        draw = PIL.ImageDraw.Draw(img)
        draw.rectangle(
            [origin, (origin[0] + width, origin[1] + height)],
            fill=color, # This fills the rectangle, outline is for border
            outline=color if thickness > 0 else None, # Use outline for border
            width=thickness,
        )
        return img

    @staticmethod
    def draw_text(
        img: PIL.Image.Image,
        origin: Tuple[int, int],
        text: str,
        font: PIL.ImageFont.FreeTypeFont = None,
        color: Union[str, Tuple[int, int, int]] = (255, 255, 255), # RGB
    ) -> PIL.Image.Image:
        draw = PIL.ImageDraw.Draw(img)
        font_to_use = font or DEFAULT_FONT_REGULAR_20
        draw.text(origin, text, font=font_to_use, fill=color)
        return img

    @staticmethod
    def draw_bounding_box(
        img: PIL.Image.Image, 
        rectangle_points: Tuple[Tuple[int, int], Tuple[int, int]], # ((xmin, ymin), (xmax, ymax))
        color: Tuple[int, int, int], 
        thickness: int = 3,
        radius: int = 7
    ) -> PIL.Image.Image:
        draw = PIL.ImageDraw.Draw(img)
        # Ensure points are tuples for rounded_rectangle
        p1 = tuple(map(int, rectangle_points[0]))
        p2 = tuple(map(int, rectangle_points[1]))
        draw.rounded_rectangle([p1, p2], radius=radius, outline=color, width=thickness)
        return img

    @staticmethod
    def draw_detection(
        detection: norfair.Detection,
        img: PIL.Image.Image,
        confidence: bool = False,
        id_label: bool = False, # Renamed from id to avoid conflict with builtin
    ) -> PIL.Image.Image:
        if detection is None or detection.points is None:
            return img

        # Ensure points are in the correct format for drawing
        p1 = tuple(detection.points[0].astype(int))
        p2 = tuple(detection.points[1].astype(int))
        
        # Use a default color if not specified, ensure it's RGB for PIL
        det_color_rgb = detection.data.get("color", (0, 0, 0)) # Black default
        if len(det_color_rgb) == 4: # RGBA
            det_color_rgb = det_color_rgb[:3] # Take RGB part

        img = Draw.draw_bounding_box(img=img, rectangle_points=(p1,p2), color=det_color_rgb)

        label_y_offset = 20
        if "label" in detection.data:
            label = detection.data["label"]
            img = Draw.draw_text(
                img=img, origin=(p1[0], p1[1] - label_y_offset), text=label, color=det_color_rgb
            )

        if id_label and "id" in detection.data:
            obj_id = detection.data["id"]
            # Adjust x for ID to be on the right, ensure text is visible
            id_text = f"ID: {obj_id}"
            # font_id = DEFAULT_FONT_REGULAR_20 # Assuming same font as label
            # id_text_width = font_id.getsize(id_text)[0] # PIL getsize deprecated, use textlength
            # id_text_width = PIL.ImageDraw.Draw(img).textlength(id_text, font=font_id)
            # origin_x_id = p2[0] - id_text_width if p2[0] - id_text_width > p1[0] else p1[0] + 5 # place on right or near left
            # Simplified: place near top-right corner of the box
            img = Draw.draw_text(
                img=img, origin=(p2[0] - 50, p1[1] - label_y_offset), text=id_text, color=det_color_rgb
            )


        if confidence and "p" in detection.data:
            conf_text = str(round(detection.data["p"], 2))
            img = Draw.draw_text(
                img=img, origin=(p1[0], p2[1] + 5), text=conf_text, color=det_color_rgb # Below the box
            )
        return img

    @staticmethod
    def draw_pointer(
        detection: norfair.Detection, img: PIL.Image.Image, color: Optional[Tuple[int,int,int]] = None
    ) -> PIL.Image.Image:
        if detection is None or detection.points is None:
            return img # Return original image if no detection

        # Default color to green if not provided
        pointer_color_rgb = color or (0, 255, 0) # Green default
        if len(pointer_color_rgb) == 4: # RGBA
            pointer_color_rgb = pointer_color_rgb[:3]

        x1, y1 = map(int, detection.points[0])
        x2, y2 = map(int, detection.points[1])

        draw = PIL.ImageDraw.Draw(img)

        width = 20
        height = 20
        vertical_space_from_bbox = 7

        t_x3 = (x1 + x2) / 2
        t_x1 = t_x3 - width / 2
        t_x2 = t_x3 + width / 2

        t_y1 = y1 - vertical_space_from_bbox - height
        t_y3 = y1 - vertical_space_from_bbox # Tip of the triangle
        # t_y2 = t_y1, already defined by y1

        triangle_points = [
            (t_x1, t_y1), (t_x2, t_y1), (t_x3, t_y3) # (t_x2, t_y2) is (t_x2, t_y1)
        ]
        
        draw.polygon(triangle_points, fill=pointer_color_rgb, outline="black") # Outline for better visibility
        # The line drawing part in original code seems redundant if polygon is outlined
        return img

    @staticmethod
    def rounded_rectangle(
        img: PIL.Image.Image, 
        rectangle_points: Tuple[Tuple[int, int], Tuple[int, int]], 
        color: Tuple[int, int, int, Optional[int]], # RGBA or RGB
        radius: int = 15,
        fill_color: bool = True # If true, fill, else just outline
    ) -> PIL.Image.Image:
        # Ensure color has alpha for overlay
        if len(color) == 3:
            color_rgba = color + (255,) # Opaque
        else:
            color_rgba = color
            
        overlay = img.copy() # Work on a copy
        draw = PIL.ImageDraw.Draw(overlay, "RGBA") # Ensure overlay is RGBA
        
        p1 = tuple(map(int, rectangle_points[0]))
        p2 = tuple(map(int, rectangle_points[1]))

        if fill_color:
            draw.rounded_rectangle([p1, p2], radius, fill=color_rgba)
        else:
            draw.rounded_rectangle([p1, p2], radius, outline=color_rgba, width=2) # Example width
        
        # Alpha composite the overlay back onto the original image
        img.alpha_composite(overlay) # This assumes img is also RGBA
        # If img is RGB, need: img.paste(overlay, (0,0), overlay)
        return img # Or return overlay if that's the workflow

    @staticmethod
    def half_rounded_rectangle(
        img: PIL.Image.Image,
        rectangle: tuple, # ((x1,y1), (x2,y2))
        color: tuple, # RGB or RGBA
        radius: int = 15,
        left: bool = False, # True if flat side is on the left, rounded on right
    ) -> PIL.Image.Image:
        
        # Ensure color has alpha for overlay drawing
        if len(color) == 3:
            color_rgba = color + (255,) # Opaque
        else:
            color_rgba = color # Assume it's already RGBA

        # Create an RGBA overlay if img is not already RGBA
        # For simplicity, assuming img can accept RGBA drawing or is converted before this stage.
        # A common pattern is to draw on an RGBA overlay then composite.
        
        # Create a transparent overlay of the same size as the image
        overlay = PIL.Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = PIL.ImageDraw.Draw(overlay) # Draw on this transparent layer

        r_x1, r_y1 = map(int, rectangle[0])
        r_x2, r_y2 = map(int, rectangle[1])
        
        # Draw the rounded rectangle part
        draw.rounded_rectangle(((r_x1, r_y1), (r_x2, r_y2)), radius, fill=color_rgba)

        # Draw a regular rectangle to cover the unwanted rounded corners on one side
        # Effectively "cutting off" the rounding.
        height = r_y2 - r_y1
        # stop_width determines how much of the rounded corner to effectively "flatten"
        # Should be at least equal to the radius.
        stop_width = radius 

        if left: # Flat on left, rounded on right
            # Cover the left rounded corners
            draw.rectangle(
                ((r_x1, r_y1), (r_x1 + stop_width, r_y2)),
                fill=color_rgba
            )
        else: # Rounded on left, flat on right
            # Cover the right rounded corners
            draw.rectangle(
                ((r_x2 - stop_width, r_y1), (r_x2, r_y2)),
                fill=color_rgba
            )
        
        # Composite the drawn overlay onto the original image
        # This requires img to be in RGBA mode or converted
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        img.alpha_composite(overlay)
        return img

    @staticmethod
    def text_in_middle_rectangle(
        img: PIL.Image.Image,
        origin: Tuple[int, int],
        width: int,
        height: int,
        text: str,
        font: PIL.ImageFont.FreeTypeFont = None,
        color: Union[str, Tuple[int, int, int]] = (255, 255, 255),
    ) -> PIL.Image.Image:
        draw = PIL.ImageDraw.Draw(img)
        font_to_use = font or DEFAULT_FONT_REGULAR_24

        # Get text bounding box using textbbox for more accuracy
        # text_bbox = draw.textbbox((0,0), text, font=font_to_use) 
        # w = text_bbox[2] - text_bbox[0]
        # h = text_bbox[3] - text_bbox[1]
        
        # Using textlength and assuming height based on font size for simplicity like original
        # For more precise vertical centering, textbbox or font metrics are better.
        text_width = draw.textlength(text, font=font_to_use)
        # Approximate text height (ascender + descender)
        # h = font_to_use.getmetrics()[0] + font_to_use.getmetrics()[1] # For older PIL
        # For newer PIL, can use textbbox as above or approximate:
        _, top, _, bottom = font_to_use.getbbox(text)
        text_height = bottom - top


        text_origin_x = origin[0] + (width - text_width) / 2
        text_origin_y = origin[1] + (height - text_height) / 2 - top # Adjust by top offset of bbox

        draw.text((text_origin_x, text_origin_y), text, font=font_to_use, fill=color)
        return img

    @staticmethod
    def add_alpha(img: PIL.Image.Image, alpha_value: int = 100) -> PIL.Image.Image:
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        alpha_channel = img.split()[-1] # Get current alpha
        
        # Create new alpha channel:
        # Where original alpha is 0 (fully transparent), keep it 0.
        # Otherwise, set to alpha_value.
        new_alpha_data = np.array(alpha_channel)
        modified_alpha_data = np.where(new_alpha_data == 0, 0, alpha_value)
        new_alpha_channel = PIL.Image.fromarray(modified_alpha_data.astype(np.uint8), mode='L')

        img.putalpha(new_alpha_channel)
        return img


class PathPoint:
    def __init__(
        self, id_val: int, center: Tuple[int, int], 
        color: Tuple[int, int, int] = (255, 255, 255), alpha: float = 1.0
    ):
        self.id: int = id_val
        self.center: Tuple[int, int] = center
        self.color: Tuple[int, int, int] = color # RGB
        self.alpha: float = np.clip(alpha, 0.0, 1.0) # Ensure alpha is [0,1]

    def __str__(self) -> str:
        return f"PathPoint(id={self.id}, center={self.center})"

    @property
    def color_with_alpha(self) -> Tuple[int, int, int, int]: # RGBA
        return self.color + (int(self.alpha * 255),)

    @staticmethod
    def get_center_from_bounding_box(bounding_box: np.ndarray) -> Optional[Tuple[int, int]]:
        """Helper to get rounded integer center from [[x1,y1],[x2,y2]] bbox."""
        center_float = get_bbox_center(bounding_box)
        return round_tuple_coords(center_float)

    @staticmethod
    def from_abs_bbox(
        id_val: int,
        abs_bbox_points: np.ndarray, # Absolute [[x1,y1],[x2,y2]]
        coord_transformations: "CoordinatesTransformation", # Forward reference
        color: Optional[Tuple[int, int, int]] = None,
        alpha: Optional[float] = None,
    ) -> Optional["PathPoint"]:
        # Convert absolute bbox to relative frame coordinates
        # Assumes coord_transformations.abs_to_rel returns [[x1,y1],[x2,y2]] in frame coords
        if coord_transformations is None: # Should not happen if types are enforced
             print("Warning: coord_transformations is None in PathPoint.from_abs_bbox")
             return None

        rel_bbox_points = coord_transformations.abs_to_rel(abs_bbox_points)
        if rel_bbox_points is None:
            return None
            
        center = PathPoint.get_center_from_bounding_box(rel_bbox_points)
        if center is None:
            return None

        default_color = (255, 255, 255) # White
        default_alpha = 1.0
        
        return PathPoint(
            id_val=id_val, 
            center=center, 
            color=color if color is not None else default_color, 
            alpha=alpha if alpha is not None else default_alpha
        )


class AbsolutePath:
    # Configuration for drawing paths
    PATH_THICKNESS = 4
    ARROW_THICKNESS = 4
    ARROW_HEAD_LENGTH = 10
    ARROW_HEAD_HEIGHT = 6
    ARROW_FRAME_FREQUENCY = 30  # Draw arrow every N points
    ARROW_LOOKBACK_FRAMES = 4   # Use point from N frames ago for arrow direction
    ALPHA_DECAY_FACTOR = 1.2    # For fading older parts of the path
    FILTER_POINTS_MARGIN = 250  # Margin for filtering points outside frame

    def __init__(self) -> None:
        # Stores tuples of (absolute_points_bbox, color)
        self.past_points_data: List[Tuple[np.ndarray, Tuple[int,int,int]]] = []

    @property
    def path_length(self) -> int:
        return len(self.past_points_data)

    def _draw_path_segment(
        self,
        draw: PIL.ImageDraw.ImageDraw,
        p1: PathPoint,
        p2: PathPoint,
        thickness: int,
    ):
        # Use p1's color and alpha for the segment leading to p2
        draw.line([p1.center, p2.center], fill=p1.color_with_alpha, width=thickness)

    def _draw_arrow_head(
        self,
        draw: PIL.ImageDraw.ImageDraw,
        start_coords: Tuple[int, int],
        end_coords: Tuple[int, int],
        color_rgba: Tuple[int, int, int, int],
        length: int,
        height: int,
        thickness: int,
    ):
        dX = end_coords[0] - start_coords[0]
        dY = end_coords[1] - start_coords[1]
        vec_len = sqrt(dX*dX + dY*dY)
        if vec_len == 0: return

        udX, udY = dX / vec_len, dY / vec_len  # Normalized direction
        perpX, perpY = -udY, udX              # Perpendicular vector

        left_x = end_coords[0] - length * udX + height * perpX
        left_y = end_coords[1] - length * udY + height * perpY
        right_x = end_coords[0] - length * udX - height * perpX
        right_y = end_coords[1] - length * udY - height * perpY
        
        draw.line([(left_x, left_y), end_coords], fill=color_rgba, width=thickness)
        draw.line([(right_x, right_y), end_coords], fill=color_rgba, width=thickness)

    def _filter_points_outside_frame(
        self, path_points: List[PathPoint], frame_width: int, frame_height: int, margin: int = 0
    ) -> List[PathPoint]:
        return [
            p for p in path_points
            if (0 - margin) < p.center[0] < (frame_width + margin) and \
               (0 - margin) < p.center[1] < (frame_height + margin)
        ]

    def add_new_point(self, detection: norfair.Detection, color: Tuple[int,int,int] = (255, 255, 255)):
        if detection is None or detection.absolute_points is None:
            return
        self.past_points_data.append((detection.absolute_points.copy(), color))


    def draw(
        self,
        img: PIL.Image.Image, # Expects RGBA image for alpha blending
        current_detection: Optional[norfair.Detection],
        coord_transformations: "CoordinatesTransformation",
        color: Tuple[int, int, int] = (255, 255, 255),
    ) -> PIL.Image.Image:
        if current_detection:
            self.add_new_point(detection=current_detection, color=color)

        if self.path_length < 2:
            return img

        # Create PathPoint objects from stored absolute data
        path_points_rel: List[PathPoint] = []
        for i, (abs_bbox, point_color) in enumerate(self.past_points_data):
            alpha = max(0.0, 1.0 - (self.path_length - 1 - i) / (self.ALPHA_DECAY_FACTOR * self.path_length)) if self.path_length > 1 else 1.0
            #alpha = i / (self.ALPHA_DECAY_FACTOR * self.path_length) if self.path_length > 0 else 1.0
            
            pp = PathPoint.from_abs_bbox(
                id_val=i,
                abs_bbox_points=abs_bbox,
                coord_transformations=coord_transformations,
                color=point_color,
                alpha=alpha
            )
            if pp:
                path_points_rel.append(pp)
        
        if not path_points_rel or len(path_points_rel) < 2:
            return img

        # Filter points outside the frame (with margin) for drawing efficiency
        # Drawing still uses all points for arrows if needed, filtering is for path segments
        # For path drawing, only draw segments where at least one point is somewhat visible
        # This is a complex optimization. Simpler: filter, then draw.
        # path_to_draw = self._filter_points_outside_frame(
        #     path_points_rel, img.width, img.height, self.FILTER_POINTS_MARGIN
        # )
        # If path_to_draw is used, ensure indices for arrows are still valid.

        # Create a drawing context on an RGBA version of the image
        # Or ensure img is already RGBA
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        draw_context = PIL.ImageDraw.Draw(img, "RGBA")

        # Draw path segments
        for i in range(len(path_points_rel) - 1):
            self._draw_path_segment(draw_context, path_points_rel[i], path_points_rel[i+1], self.PATH_THICKNESS)

        # Draw arrows
        for i, point in enumerate(path_points_rel):
            if i < self.ARROW_LOOKBACK_FRAMES or i % self.ARROW_FRAME_FREQUENCY != 0:
                continue
            
            start_point = path_points_rel[i - self.ARROW_LOOKBACK_FRAMES]
            end_point = point # current point is the arrow tip
            
            self._draw_arrow_head(
                draw_context,
                start_point.center,
                end_point.center,
                start_point.color_with_alpha, # Color of arrow based on starting segment
                self.ARROW_HEAD_LENGTH,
                self.ARROW_HEAD_HEIGHT,
                self.ARROW_THICKNESS
            )
        return img