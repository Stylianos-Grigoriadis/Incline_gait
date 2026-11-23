"""Fit an anatomy image to an avatar and export a transparent PNG."""

from dataclasses import dataclass, replace
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import PolygonSelector
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


PROJECT = Path(__file__).resolve().parent
IMAGE_TYPES = [("Images", "*.png *.jpg *.jpeg *.webp")]
EXPORT_DPI = 600
VM_LANDMARKS = ((0.650, 0.462), (0.569, 0.698))
RIGHT_THIGH = [(0.70, 0.455), (0.56, 0.468), (0.52, 0.56),
               (0.555, 0.665), (0.595, 0.72), (0.66, 0.72),
               (0.70, 0.665), (0.735, 0.56)]
STERNUM = [(0.42, 0.218), (0.58, 0.218), (0.65, 0.25),
           (0.65, 0.302), (0.60, 0.341), (0.50, 0.365),
           (0.40, 0.341), (0.35, 0.302), (0.35, 0.25)]
REAR_MEDIAL_CALF = [(0.615, 0.696), (0.646, 0.720), (0.650, 0.761),
                    (0.631, 0.801), (0.615, 0.854), (0.596, 0.849),
                    (0.572, 0.795), (0.564, 0.760), (0.578, 0.722)]


@dataclass(eq=False)
class MovableImage:
    source: Image.Image
    center: tuple[float, float]
    width: float
    angle: float = 0

    def render(self):
        scale = self.width / self.source.width
        size = (max(1, round(self.width)), max(1, round(self.source.height * scale)))
        return self.source.resize(size, Image.Resampling.LANCZOS).rotate(
            self.angle, resample=Image.Resampling.BICUBIC, expand=True,
        )


def draw_vastus_medialis_guide(image):
    """Draw the approximate SENIAM reference line on the front-view avatar."""
    guide = image.copy()
    draw = ImageDraw.Draw(guide)
    start, end = [(x * image.width, y * image.height) for x, y in VM_LANDMARKS]
    line_width = max(1, round(image.width * 0.005))
    radius = max(2, round(image.width * 0.010))
    for dash in range(14):
        begin = dash / 14
        finish = (dash + 0.58) / 14
        segment = [tuple(a + (b - a) * fraction for a, b in zip(start, end))
                   for fraction in (begin, finish)]
        draw.line(segment, fill="black", width=line_width)
    for x, y in (start, end):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="black")
    return guide


def compose(avatar, anatomy, opacity, right_thigh, offset, scale, *, sternum=False,
            gastrocnemius=False, custom_regions=()):
    """Blend an aligned anatomy layer while retaining the avatar's alpha."""
    width, height = avatar.size
    fitted = anatomy.resize((max(1, round(width * scale)), max(1, round(height * scale))),
                            Image.Resampling.LANCZOS)
    position = tuple(round((base - adjusted) / 2 + shift)
                     for base, adjusted, shift in zip(avatar.size, fitted.size, offset))
    layer = Image.new("RGBA", avatar.size)
    layer.paste(fitted, position)
    mask = layer.getchannel("A")
    if right_thigh or sternum or gastrocnemius or custom_regions:
        region = Image.new("L", avatar.size)
        feather = ImageFilter.GaussianBlur(width * 0.012)
        if right_thigh:
            ImageDraw.Draw(region).polygon([(x * width, y * height) for x, y in RIGHT_THIGH], fill=255)
            region = region.filter(feather)
            region.paste(0, (0, 0, (width + 1) // 2, height))
        if sternum:
            chest = Image.new("L", avatar.size)
            ImageDraw.Draw(chest).polygon([(x * width, y * height) for x, y in STERNUM], fill=255)
            region = ImageChops.lighter(region, chest.filter(feather))
        if gastrocnemius:
            calf = Image.new("L", avatar.size)
            ImageDraw.Draw(calf).polygon([(x * width, y * height) for x, y in REAR_MEDIAL_CALF], fill=255)
            region = ImageChops.lighter(region, calf.filter(feather))
        if custom_regions:
            custom = Image.new("L", avatar.size)
            draw = ImageDraw.Draw(custom)
            for vertices in custom_regions:
                draw.polygon(vertices, fill=255)
            region = ImageChops.lighter(region, custom.filter(feather))
        mask = ImageChops.multiply(mask, region)
    mask = mask.point(lambda pixel: round(pixel * opacity))
    result = Image.composite(layer.convert("RGB"), avatar.convert("RGB"), mask)
    result.putalpha(avatar.getchannel("A"))
    return result


class AvatarEditor:
    def __init__(self, root):
        self.root = root
        self.avatar = self.result = None
        self.muscles = []
        self.custom_regions = []
        self.active_view = "Front"
        self.view_images = {}
        self.movable_images = {}
        self.movable_artists = {}
        self.selected_image = None
        self.copied_image = None
        self.drag_origin = None
        self.view = tk.StringVar(value=self.active_view)
        self.selector = None
        self.opacity = tk.DoubleVar(value=75)
        self.scale = tk.DoubleVar(value=100)
        self.horizontal = tk.DoubleVar()
        self.vertical = tk.DoubleVar()
        self.right_thigh = tk.BooleanVar(value=False)
        self.sternum = tk.BooleanVar(value=False)
        self.vastus_medialis = tk.BooleanVar(value=False)
        self.gastrocnemius = tk.BooleanVar(value=True)

        root.title("Avatar anatomy overlay")
        controls = ttk.Frame(root, padding=16)
        controls.pack(side="left", fill="y")
        view_picker = ttk.Combobox(controls, textvariable=self.view, values=("Front", "Rear"),
                                   state="readonly")
        view_picker.pack(fill="x", pady=(0, 6))
        view_picker.bind("<<ComboboxSelected>>", self.switch_view)
        ttk.Button(controls, text="Open avatar image", command=self.open_avatar).pack(fill="x")
        ttk.Button(controls, text="Add muscle PNGs", command=self.add_muscles).pack(fill="x")
        self.layer_controls = ttk.LabelFrame(controls, text="Visible muscles", padding=6)
        self.layer_controls.pack(fill="x", pady=(10, 0))
        regions = ttk.LabelFrame(controls, text="Cutaway regions", padding=6)
        regions.pack(fill="x", pady=10)
        ttk.Checkbutton(regions, text="Viewer-right thigh", variable=self.right_thigh,
                        command=self.refresh).pack(anchor="w")
        ttk.Checkbutton(regions, text="Sternum and nearby ribs", variable=self.sternum,
                        command=self.refresh).pack(anchor="w")
        ttk.Checkbutton(regions, text="Rear gastrocnemius medialis", variable=self.gastrocnemius,
                        command=self.refresh).pack(anchor="w")
        ttk.Button(regions, text="Draw custom cutaway", command=self.draw_cutaway).pack(fill="x")
        ttk.Button(regions, text="Clear custom cutaways", command=self.clear_cutaway).pack(fill="x")
        ttk.Checkbutton(controls, text="Vastus medialis guide", variable=self.vastus_medialis,
                        command=self.refresh).pack(anchor="w", pady=(0, 10))
        ttk.Button(controls, text="Add movable image", command=self.open_movable_image).pack(fill="x")
        image_actions = ttk.Frame(controls)
        image_actions.pack(fill="x")
        for label, action in (("Copy", self.copy_image), ("Paste", self.paste_image),
                              ("Remove", self.remove_movable_image)):
            ttk.Button(image_actions, text=label, command=action).pack(side="left", expand=True, fill="x")
        ttk.Label(controls, text="Drag to move · Scroll to resize\nShift + scroll to rotate",
                  justify="left").pack(anchor="w", pady=(2, 8))
        for label, variable, low, high in [
            ("Anatomy visibility", self.opacity, 0, 100),
            ("Anatomy scale (%)", self.scale, 30, 170),
            ("Horizontal shift (% of width)", self.horizontal, -50, 50),
            ("Vertical shift (% of height)", self.vertical, -50, 50),
        ]:
            ttk.Label(controls, text=label).pack(anchor="w")
            self.add_slider(controls, variable, low, high)
        ttk.Button(controls, text=f"Save PNG at {EXPORT_DPI} DPI", command=self.save).pack(fill="x", pady=10)
        self.status = ttk.Label(controls, text="Open the original avatar to begin.", wraplength=220)
        self.status.pack(anchor="w")
        self.figure = Figure(figsize=(4.5, 6), layout="tight")
        self.axis = self.figure.add_subplot()
        self.axis.set_axis_off()
        self.axis.set_autoscale_on(False)
        self.plot_image = None
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(side="right", expand=True, fill="both")
        self.canvas.mpl_connect("button_press_event", self.start_image_drag)
        self.canvas.mpl_connect("motion_notify_event", self.move_image)
        self.canvas.mpl_connect("button_release_event", self.end_image_drag)
        self.canvas.mpl_connect("scroll_event", self.adjust_image)
        self.canvas.mpl_connect("key_press_event", self.image_shortcut)

    @staticmethod
    def read_image(path):
        with Image.open(path) as image:
            return image.convert("RGBA")

    def add_slider(self, parent, variable, low, high):
        percentage = tk.StringVar(value=f"{variable.get():g}")

        def slider_changed(_value):
            percentage.set(f"{variable.get():.6g}")
            self.refresh()

        def apply_percentage(_event):
            try:
                value = float(percentage.get().strip().removesuffix("%"))
            except ValueError:
                value = variable.get()
            if low <= value <= high:
                variable.set(value)
            slider_changed(None)

        ttk.Scale(parent, from_=low, to=high, variable=variable,
                  command=slider_changed).pack(fill="x")
        entry = ttk.Entry(parent, textvariable=percentage, width=9)
        entry.pack(anchor="w", pady=(2, 10))
        entry.bind("<Return>", apply_percentage)
        entry.bind("<FocusOut>", apply_percentage)

    def open_avatar(self):
        path = filedialog.askopenfilename(title="Open avatar image", filetypes=IMAGE_TYPES)
        if path:
            try:
                self.avatar = self.read_image(path)
            except OSError as error:
                messagebox.showerror("Could not open image", str(error))
                return
            self.clear_cutaway()

    def open_movable_image(self):
        if self.avatar is None:
            messagebox.showinfo("Open an avatar", "Load an avatar before adding a movable image.")
            return
        path = filedialog.askopenfilename(title="Open movable image", filetypes=IMAGE_TYPES)
        if not path:
            return
        try:
            image = self.read_image(path)
        except OSError as error:
            messagebox.showerror("Could not open image", str(error))
            return
        width, height = self.avatar.size
        initial_width = min(width * 0.18, height * 0.18 * image.width / image.height)
        self.stop_drawing()
        self.end_image_drag(None)
        self.add_movable_image(MovableImage(image, (width / 2, height / 2), initial_width))

    def add_movable_image(self, image):
        self.movable_images.setdefault(self.active_view, []).append(image)
        self.selected_image = image
        self.refresh()

    def copy_image(self):
        if self.selected_image is not None:
            self.copied_image = replace(self.selected_image)

    def paste_image(self):
        if self.avatar is not None and self.copied_image is not None:
            self.stop_drawing()
            self.end_image_drag(None)
            self.add_movable_image(replace(self.copied_image))

    def image_shortcut(self, event):
        if event.key == "ctrl+c":
            self.copy_image()
        elif event.key == "ctrl+v":
            self.paste_image()

    def remove_movable_image(self):
        if self.selected_image is None:
            return
        self.movable_images[self.active_view].remove(self.selected_image)
        self.selected_image = None
        self.end_image_drag(None)
        self.refresh()

    def image_hit(self, event):
        if self.selector is not None or event.inaxes is not self.axis:
            return None
        for image in reversed(self.movable_images.get(self.active_view, [])):
            left, right, bottom, top = self.movable_artists[image].get_extent()
            if left <= event.xdata <= right and top <= event.ydata <= bottom:
                return image
        return None

    def start_image_drag(self, event):
        image = self.image_hit(event)
        if event.button == MouseButton.LEFT and image is not None:
            self.selected_image = image
            self.canvas.get_tk_widget().focus_set()
            self.drag_origin = (event.xdata - image.center[0], event.ydata - image.center[1])

    def move_image(self, event):
        if self.drag_origin is None or event.inaxes is not self.axis:
            return
        image = self.selected_image
        image.center = (event.xdata - self.drag_origin[0], event.ydata - self.drag_origin[1])
        self.update_movable_artist(image)

    def end_image_drag(self, _event):
        self.drag_origin = None

    def adjust_image(self, event):
        image = self.image_hit(event)
        if image is None:
            return
        self.selected_image = image
        if "shift" in getattr(event, "modifiers", ()) or event.key == "shift":
            image.angle = (image.angle + event.step * 5) % 360
        else:
            maximum = 2 * min(self.avatar.width, self.avatar.height * image.source.width / image.source.height)
            image.width = min(maximum, max(1, image.width * 1.12 ** event.step))
        self.update_movable_artist(image, render=True)

    def update_movable_artist(self, image, *, render=False):
        artist = self.movable_artists.get(image)
        if artist is None:
            artist = self.axis.imshow(image.render(), zorder=3)
            self.movable_artists[image] = artist
        elif render:
            artist.set_data(image.render())
        height, width = artist.get_array().shape[:2]
        x, y = image.center
        artist.set_extent((x - width / 2, x + width / 2, y + height / 2, y - height / 2))
        self.canvas.draw_idle()

    def switch_view(self, _event):
        selected = self.view.get()
        if selected == self.active_view:
            return
        self.stop_drawing()
        self.end_image_drag(None)
        self.view_images[self.active_view] = (self.avatar, self.muscles, self.custom_regions)
        self.active_view = selected
        self.selected_image = None
        self.avatar, self.muscles, self.custom_regions = self.view_images.get(selected, (None, [], []))
        if selected == "Rear" and self.avatar is None:
            try:
                rear = self.read_image(PROJECT / "Avatar_back_anatomy.png")
            except OSError as error:
                messagebox.showerror("Could not load rear anatomy", str(error))
            else:
                self.avatar = ImageOps.grayscale(rear).convert("RGBA")
                self.avatar.putalpha(rear.getchannel("A"))
                self.muscles.append(("Rear anatomy", rear, tk.BooleanVar(value=True)))
        for checkbox in self.layer_controls.winfo_children():
            checkbox.destroy()
        for label, _, visible in self.muscles:
            self.add_layer_control(label, visible)
        self.refresh()

    def draw_cutaway(self):
        if self.avatar is None:
            messagebox.showinfo("Open an avatar", "Load an avatar before drawing a cutaway.")
            return
        self.stop_drawing()
        self.end_image_drag(None)
        self.selector = PolygonSelector(self.axis, self.finish_cutaway, useblit=False)
        self.status.configure(text="Click around the area. Click the first point to close the outline.")

    def stop_drawing(self):
        if self.selector is not None:
            self.selector.set_active(False)
            self.selector.set_visible(False)
            self.selector.disconnect_events()
            self.selector = None

    def finish_cutaway(self, vertices):
        self.custom_regions.append(list(vertices))
        self.stop_drawing()
        self.refresh()

    def clear_cutaway(self):
        self.stop_drawing()
        self.custom_regions.clear()
        self.refresh()

    def add_muscles(self):
        paths = filedialog.askopenfilenames(
            title="Choose one transparent PNG per muscle", initialdir=PROJECT,
            filetypes=[("Transparent PNG", "*.png")],
        )
        for path in paths:
            try:
                image = self.read_image(path)
            except OSError as error:
                messagebox.showerror("Could not open image", str(error))
                continue
            if image.getchannel("A").getextrema()[0] == 255:
                messagebox.showerror("Transparency required", f"{Path(path).name} has no transparent background.")
                continue
            visible = tk.BooleanVar(value=True)
            label = Path(path).stem.replace("_", " ")
            self.muscles.append((label, image, visible))
            self.add_layer_control(label, visible)
        self.refresh()

    def add_layer_control(self, label, visible):
        ttk.Checkbutton(self.layer_controls, text=label, variable=visible,
                        command=self.refresh).pack(anchor="w")

    def refresh(self, _value=None):
        images = self.movable_images.get(self.active_view, []) if self.avatar is not None else []
        for image in self.movable_artists.keys() - set(images):
            self.movable_artists.pop(image).remove()
        if self.avatar is None:
            self.result = None
            if self.plot_image is not None:
                self.plot_image.remove()
                self.plot_image = None
            self.status.configure(text="Open an avatar image to begin.")
            self.canvas.draw_idle()
            return
        width, height = self.avatar.size
        self.result = self.avatar
        anatomy = Image.new("RGBA", self.avatar.size)
        for _, image, visible in self.muscles:
            if visible.get():
                anatomy.alpha_composite(image.resize(self.avatar.size, Image.Resampling.LANCZOS))
        front = self.active_view == "Front"
        if self.muscles:
            self.result = compose(self.avatar, anatomy, self.opacity.get() / 100,
                                  front and self.right_thigh.get(),
                                  (width * self.horizontal.get() / 100, height * self.vertical.get() / 100),
                                  self.scale.get() / 100, sternum=front and self.sternum.get(),
                                  gastrocnemius=not front and self.gastrocnemius.get(),
                                  custom_regions=self.custom_regions)
        if front and self.vastus_medialis.get():
            self.result = draw_vastus_medialis_guide(self.result)
        self.status.configure(text=f"Export: {width * 2} × {height * 2} px at {EXPORT_DPI} DPI")
        if self.plot_image is not None:
            self.plot_image.remove()
        self.plot_image = self.axis.imshow(self.result)
        for image in images:
            self.update_movable_artist(image)
        self.axis.set_xlim(-0.5, width - 0.5)
        self.axis.set_ylim(height - 0.5, -0.5)
        self.canvas.draw_idle()

    def save(self):
        if self.result is None:
            messagebox.showinfo("Open an avatar", "Load an avatar before exporting.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            try:
                size = tuple(dimension * 2 for dimension in self.result.size)
                exported = self.result.resize(size, Image.Resampling.LANCZOS)
                for image in self.movable_images.get(self.active_view, []):
                    overlay = MovableImage(image.source, image.center, image.width * 2, image.angle).render()
                    position = tuple(round(center * 2 - length / 2)
                                     for center, length in zip(image.center, overlay.size))
                    exported.alpha_composite(overlay, dest=position)
                exported.save(path, format="PNG", dpi=(EXPORT_DPI, EXPORT_DPI))
            except OSError as error:
                messagebox.showerror("Could not save PNG", str(error))


if __name__ == "__main__":
    window = tk.Tk()
    AvatarEditor(window)
    window.mainloop()
