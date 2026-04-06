"""HUD overlay rendering methods for the interactive viewer."""

import queue

import numpy as np


class HUDRenderer:
    """Methods for title, legend, and help text overlay rendering.

    Accesses viewer state via ``self.v`` (back-reference to InteractiveViewer).
    """

    def __init__(self, viewer):
        self.v = viewer

    # ------------------------------------------------------------------
    # Title & legend overlays
    # ------------------------------------------------------------------

    def _render_title_overlay(self):
        """Pre-render title + subtitle with drop-shadow to an RGBA array.

        Rendered at 2x resolution for antialiased text, then downsampled.
        No background — text floats over the scene with a strong shadow.
        """
        v = self.v
        if not v._title or v._title == 'rtxpy':
            return
        try:
            from PIL import Image, ImageDraw, ImageFont, ImageFilter

            bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            sans_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

            # Render at 2x for antialiasing via supersample
            ss = 2
            try:
                font_title = ImageFont.truetype(bold_path, 22 * ss)
                font_sub = ImageFont.truetype(sans_path, 11 * ss)
            except (OSError, IOError):
                font_title = ImageFont.load_default()
                font_sub = font_title
                ss = 1  # fall back to 1x with default font

            pad_x, pad_y = 14 * ss, 10 * ss
            shadow_spread = 4 * ss  # extra margin for shadow blur

            # Measure title
            title_bbox = font_title.getbbox(v._title)
            title_w = title_bbox[2] - title_bbox[0]
            title_h = title_bbox[3] - title_bbox[1]

            # Measure subtitle
            sub_h = 0
            sub_w = 0
            if v._subtitle:
                sub_bbox = font_sub.getbbox(v._subtitle)
                sub_w = sub_bbox[2] - sub_bbox[0]
                sub_h = sub_bbox[3] - sub_bbox[1] + 4 * ss

            img_w = pad_x * 2 + max(title_w, sub_w) + shadow_spread * 2
            img_h = pad_y * 2 + title_h + sub_h + shadow_spread * 2

            # --- Shadow layer: black text, then blur ---
            shadow_img = Image.new('RGBA', (img_w, img_h), (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow_img)
            sx = pad_x + shadow_spread
            sy = pad_y + shadow_spread

            # Draw shadow text with slight offset
            shadow_off = 2 * ss
            shadow_draw.text((sx + shadow_off, sy + shadow_off),
                             v._title, fill=(0, 0, 0, 255),
                             font=font_title)
            if v._subtitle:
                shadow_draw.text(
                    (sx + shadow_off, sy + title_h + 4 * ss + shadow_off),
                    v._subtitle, fill=(0, 0, 0, 255), font=font_sub)

            # Multi-pass blur for a deep, soft shadow
            for _ in range(3):
                shadow_img = shadow_img.filter(
                    ImageFilter.GaussianBlur(radius=3 * ss))

            # Boost shadow opacity
            shadow_arr = np.array(shadow_img, dtype=np.float32)
            shadow_arr[:, :, 3] = np.clip(shadow_arr[:, :, 3] * 2.5, 0, 255)
            shadow_img = Image.fromarray(shadow_arr.astype(np.uint8))

            # --- Foreground layer: crisp white text ---
            fg_img = Image.new('RGBA', (img_w, img_h), (0, 0, 0, 0))
            fg_draw = ImageDraw.Draw(fg_img)
            fg_draw.text((sx, sy), v._title,
                         fill=(255, 255, 255, 255), font=font_title)
            if v._subtitle:
                fg_draw.text((sx, sy + title_h + 4 * ss), v._subtitle,
                             fill=(200, 210, 225, 230), font=font_sub)

            # Composite: shadow behind, text on top
            result = Image.alpha_composite(shadow_img, fg_img)

            # Downsample from 2x to 1x with antialiasing
            if ss > 1:
                final_w = img_w // ss
                final_h = img_h // ss
                result = result.resize((final_w, final_h), Image.LANCZOS)

            v._title_overlay_rgba = (
                np.array(result, dtype=np.float32) / 255.0)
        except ImportError:
            pass

    def _render_legend_overlay(self):
        """Pre-render legend to an RGBA numpy array using PIL.

        Called once at startup; cached in ``self._legend_rgba``.
        Expects ``self._legend_config`` to be a dict with 'title' and
        'entries' keys.
        """
        v = self.v
        if not v._legend_config:
            return
        entries = v._legend_config.get('entries', [])
        if not entries:
            return
        try:
            from PIL import Image, ImageDraw, ImageFont

            bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            sans_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            try:
                font_header = ImageFont.truetype(bold_path, 13)
                font_label = ImageFont.truetype(sans_path, 12)
            except (OSError, IOError):
                font_header = ImageFont.load_default()
                font_label = font_header

            pad_x, pad_y = 14, 10
            swatch_size = 12
            swatch_gap = 8   # gap between swatch and label
            line_h = 18
            bg_color = (15, 18, 24, 200)
            header_color = (180, 210, 255, 255)
            label_color = (210, 215, 225, 220)

            legend_title = v._legend_config.get('title', '')

            # Measure widths
            max_label_w = 0
            for label, _color in entries:
                bbox = font_label.getbbox(label)
                max_label_w = max(max_label_w, bbox[2] - bbox[0])

            content_w = swatch_size + swatch_gap + max_label_w
            if legend_title:
                hdr_bbox = font_header.getbbox(legend_title)
                content_w = max(content_w, hdr_bbox[2] - hdr_bbox[0])

            img_w = pad_x * 2 + content_w
            header_h = 0
            if legend_title:
                header_h = 13 + 8  # font size + gap below header
            img_h = pad_y * 2 + header_h + len(entries) * line_h

            img = Image.new('RGBA', (img_w, img_h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.rectangle(
                [0, 0, img_w - 1, img_h - 1],
                fill=bg_color,
            )
            draw.rectangle(
                [0, 0, img_w - 1, img_h - 1],
                outline=(60, 70, 90, 140), width=1,
            )

            y = pad_y
            if legend_title:
                draw.text((pad_x, y), legend_title,
                          fill=header_color, font=font_header)
                y += header_h

            for label, color_rgb in entries:
                # Colour swatch
                r8 = int(min(1.0, color_rgb[0]) * 255)
                g8 = int(min(1.0, color_rgb[1]) * 255)
                b8 = int(min(1.0, color_rgb[2]) * 255)
                sx = pad_x
                sy = y + (line_h - swatch_size) // 2
                draw.rectangle(
                    [sx, sy, sx + swatch_size - 1, sy + swatch_size - 1],
                    fill=(r8, g8, b8, 255),
                )
                # Label
                draw.text((pad_x + swatch_size + swatch_gap, y + 1),
                          label, fill=label_color, font=font_label)
                y += line_h

            v._legend_rgba = np.array(img, dtype=np.float32) / 255.0
        except ImportError:
            pass

    def _blit_title_on_frame(self, img):
        """Alpha-composite cached title overlay onto the rendered frame.

        Placed flush in the top-left corner (no margin).

        Parameters
        ----------
        img : ndarray, shape (H, W, 3), float32 0-1
            Rendered frame. Modified in-place.
        """
        v = self.v
        if v._title_overlay_rgba is None:
            return
        ov = v._title_overlay_rgba
        oh, ow = ov.shape[:2]
        fh, fw = img.shape[:2]
        bh = min(oh, fh)
        bw = min(ow, fw)
        if bh <= 0 or bw <= 0:
            return
        alpha = ov[:bh, :bw, 3:4]
        rgb = ov[:bh, :bw, :3]
        region = img[:bh, :bw]
        region[:] = region * (1 - alpha) + rgb * alpha

    def _blit_legend_on_frame(self, img):
        """Alpha-composite cached legend overlay onto the rendered frame.

        Placed flush in the bottom-left corner (no margin).

        Parameters
        ----------
        img : ndarray, shape (H, W, 3), float32 0-1
            Rendered frame. Modified in-place.
        """
        v = self.v
        if v._legend_rgba is None:
            return
        ov = v._legend_rgba
        oh, ow = ov.shape[:2]
        fh, fw = img.shape[:2]
        y0 = fh - oh
        if y0 < 0:
            y0 = 0
        bh = min(oh, fh - y0)
        bw = min(ow, fw)
        if bh <= 0 or bw <= 0:
            return
        alpha = ov[:bh, :bw, 3:4]
        rgb = ov[:bh, :bw, :3]
        region = img[y0:y0+bh, :bw]
        region[:] = region * (1 - alpha) + rgb * alpha

    def _render_help_text(self):
        """Pre-render paginated help overlays cached in ``_help_pages``.

        Each page matches the legend/minimap height and is positioned
        at the bottom of the frame.  Press H to cycle pages, then off.
        """
        v = self.v
        all_sections = [
            ("MOVEMENT", [
                ("W/S/A/D", "Move fwd / back / left / right"),
                ("Arrows", "Move fwd / back / left / right"),
                ("Q / E", "Move up / down"),
                ("I/J/K/L", "Look up / left / down / right"),
                ("Drag", "Pan (slippy-map)"),
                ("Scroll", "Zoom (FOV)"),
                ("+ / -", "Speed up / down"),
            ]),
            ("TERRAIN", [
                ("G", "Cycle terrain layer"),
                ("U / Shift+U", "Cycle basemap fwd / back"),
                ("C", "Cycle colormap"),
                ("Y", "Cycle color stretch"),
                (", / .", "Overlay alpha"),
                ("R / Shift+R", "Resolution down / up"),
                ("Z / Shift+Z", "Vert. exag. down / up"),
                ("B", "(reserved)"),
                ("T", "Toggle shadows"),
                ("Shift+T", "Cycle time of day"),
                ("Shift+E", "Toggle terrain"),
            ]),
            ("DATA LAYERS", [
                ("Shift+F", "FIRMS fire (7d)"),
                ("Shift+W", "Toggle wind"),
                ("Shift+N", "Toggle clouds + rain"),
                ("Shift+Y", "Toggle hydro flow"),
            ]),
            ("RENDERING", [
                ("0", "Toggle ambient occlusion"),
                ("Shift+H", "Prev help page"),
                ("Shift+G", "Cycle GI bounces (1-3)"),
                ("Shift+D", "Toggle AI denoiser"),
                ("9", "Toggle depth of field"),
                ("; / '", "DOF aperture down / up"),
                (": / \"", "DOF focal dist. down / up"),
            ]),
            ("GEOMETRY", [
                ("N", "Cycle geometry layer"),
                ("P", "Prev geometry in group"),
                ("Shift+C", "Cycle point cloud colors"),
                ("Right-Click", "Pick geometry"),
            ]),
            ("OBSERVERS", [
                ("1-8", "Select / create observer"),
                ("O", "Move observer to camera"),
                ("Shift+O", "Drone mode (3rd / FPV)"),
                ("Shift+V", "Snap camera to observer"),
                ("Shift+K", "Kill all observers"),
                ("V", "Toggle viewshed"),
                ("[ / ]", "Observer height down / up"),
            ]),
            ("OTHER", [
                ("F", "Screenshot"),
                ("M", "Toggle minimap"),
                ("H", "Cycle help pages"),
                ("X / Esc", "Exit"),
            ]),
        ]

        # Append info text as a section if provided
        if v._info_text:
            info_items = []
            for line in v._info_text.strip().splitlines():
                line = line.strip()
                if line:
                    info_items.append(("", line))
            if info_items:
                all_sections.append(("MODEL INFO", info_items))

        try:
            from PIL import Image, ImageDraw, ImageFont

            font_size = 12
            header_size = 13
            mono_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
            sans_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            sans_bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            try:
                font = ImageFont.truetype(sans_path, font_size)
                font_key = ImageFont.truetype(mono_path, font_size)
                font_header = ImageFont.truetype(sans_bold_path, header_size)
            except (OSError, IOError):
                font = ImageFont.load_default()
                font_key = font
                font_header = font

            line_h = font_size + 5
            hdr_h = header_size + 8
            section_gap = 6
            key_col_w = 105
            desc_col_w = 195
            col_w = key_col_w + desc_col_w
            pad_x = 14
            pad_y = 10

            # Colors
            bg_color = (15, 18, 24, 210)
            header_color = (180, 210, 255, 255)
            key_color = (255, 200, 100, 245)
            desc_color = (210, 215, 225, 220)
            accent_color = (90, 140, 220, 180)
            page_color = (120, 140, 170, 180)

            # Target height: match legend if available
            if v._legend_rgba is not None:
                target_h = v._legend_rgba.shape[0]
            else:
                target_h = 200
            usable_h = target_h - pad_y * 2

            def _section_h(section):
                _, items = section
                return hdr_h + 3 + len(items) * line_h

            # Pack sections into pages greedily
            pages = []
            current_page = []
            current_h = 0
            for sec in all_sections:
                sh = _section_h(sec)
                needed = sh + (section_gap if current_page else 0)
                if current_page and current_h + needed > usable_h:
                    pages.append(current_page)
                    current_page = [sec]
                    current_h = sh
                else:
                    current_h += needed
                    current_page.append(sec)
            if current_page:
                pages.append(current_page)

            n_pages = len(pages)
            img_w = pad_x * 2 + col_w

            v._help_pages = []
            for pi, page_sections in enumerate(pages):
                img = Image.new('RGBA', (img_w, target_h), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                draw.rectangle(
                    [0, 0, img_w - 1, target_h - 1], fill=bg_color)
                draw.rectangle(
                    [0, 0, img_w - 1, target_h - 1],
                    outline=(60, 70, 90, 140), width=1)

                y = pad_y
                for si, (title, items) in enumerate(page_sections):
                    if si > 0:
                        y += section_gap
                    draw.text((pad_x, y), title, fill=header_color,
                              font=font_header)
                    underline_y = y + header_size + 2
                    draw.line(
                        [(pad_x, underline_y),
                         (pad_x + col_w - 10, underline_y)],
                        fill=accent_color, width=1)
                    y = underline_y + 3
                    for key_text, desc_text in items:
                        if key_text:
                            draw.text((pad_x, y), key_text,
                                      fill=key_color, font=font_key)
                            draw.text((pad_x + key_col_w, y), desc_text,
                                      fill=desc_color, font=font)
                        else:
                            draw.text((pad_x, y), desc_text,
                                      fill=desc_color, font=font)
                        y += line_h

                # Page indicator (top-right)
                page_text = f"{pi + 1}/{n_pages}"
                bbox = font_header.getbbox(page_text)
                ptw = bbox[2] - bbox[0]
                draw.text((img_w - pad_x - ptw, pad_y), page_text,
                          fill=page_color, font=font_header)

                v._help_pages.append(
                    np.array(img, dtype=np.float32) / 255.0)
        except ImportError:
            v._help_pages = []

    def _blit_help_on_frame(self, img):
        """Alpha-composite the current help page onto the rendered frame.

        Positioned flush at the bottom, right after the legend.

        Parameters
        ----------
        img : ndarray, shape (H, W, 3), float32 0-1
            Rendered frame. Modified in-place.
        """
        v = self.v
        idx = v._help_page_idx
        if idx < 0 or idx >= len(v._help_pages):
            return
        ht = v._help_pages[idx]
        hh, hw = ht.shape[:2]
        fh, fw = img.shape[:2]

        # X: right after legend, or flush left
        if v._legend_rgba is not None:
            x0 = v._legend_rgba.shape[1]
        else:
            x0 = 0

        # Flush bottom
        y0 = fh - hh
        if y0 < 0:
            y0 = 0

        bh = min(hh, fh - y0)
        bw = min(hw, fw - x0)
        if bh <= 0 or bw <= 0:
            return
        alpha = ht[:bh, :bw, 3:4]
        rgb = ht[:bh, :bw, :3]
        region = img[y0:y0+bh, x0:x0+bw]
        region[:] = region * (1 - alpha) + rgb * alpha

    def _drain_command_queue(self):
        """Drain REPL command queue — execute pending callables (thread-safe)."""
        v = self.v
        while True:
            try:
                cmd = v._command_queue.get_nowait()
            except queue.Empty:
                break
            try:
                cmd(v)
            except Exception as exc:
                import traceback
                traceback.print_exc()
            v._render_needed = True
