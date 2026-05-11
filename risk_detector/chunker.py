import io
from PIL import Image

from risk_detector.constants import CHUNK_HEIGHT, CHUNK_OVERLAP, MAX_PARSE_WIDTH, SPLIT_HEIGHT_THRESHOLD


class EstimateImageChunker:
    def prepare_chunks(self, raw: bytes) -> list[bytes]:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img = self._resize_for_parse(img)
        _, h = img.size
        if h <= SPLIT_HEIGHT_THRESHOLD:
            return [self._image_to_bytes(img)]
        return [self._image_to_bytes(c) for c in self._split_image_vertically(img)]

    def _split_image_vertically(self, img: Image.Image) -> list[Image.Image]:
        w, h = img.size
        chunks = []
        top = 0
        while top < h:
            bottom = min(top + CHUNK_HEIGHT, h)
            chunks.append(img.crop((0, top, w, bottom)))
            if bottom == h:
                break
            top += CHUNK_HEIGHT - CHUNK_OVERLAP
        return chunks

    def _resize_for_parse(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w <= MAX_PARSE_WIDTH:
            return img
        return img.resize((MAX_PARSE_WIDTH, int(h * MAX_PARSE_WIDTH / w)), Image.LANCZOS)

    def _image_to_bytes(self, img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()