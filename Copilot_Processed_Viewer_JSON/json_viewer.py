#!/usr/bin/env python3
"""
Simple viewer for browsing JSON files inside Copilot_Processed_Datasets_JSON.

Features:
- Choose any processed dataset subfolder
- Move backward/forward through JSON files
- Jump directly to a file from a dropdown
- Inspect the full pretty-printed JSON
- Browse individual fields and values
"""

import json
import os
import re
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "../Copilot_Processed_Datasets_JSON")


def natural_sort_key(value):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


class JsonViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Copilot Processed JSON Viewer")
        self.root.geometry("1500x930")
        self.root.minsize(1100, 750)

        self.dataset_folders = self.load_dataset_folders()
        self.current_folder = ""
        self.current_files = []
        self.current_index = 0
        self.current_data = {}

        self.setup_styles()
        self.setup_ui()

        if not self.dataset_folders:
            messagebox.showerror(
                "No dataset folders found",
                f"No subfolders found in {os.path.abspath(DATA_ROOT)}",
            )
            self.root.destroy()
            return

        self.folder_var.set(self.dataset_folders[0])
        self.change_folder()

    def load_dataset_folders(self):
        if not os.path.isdir(DATA_ROOT):
            return []

        folders = [
            item
            for item in os.listdir(DATA_ROOT)
            if os.path.isdir(os.path.join(DATA_ROOT, item))
        ]
        return sorted(folders, key=natural_sort_key)

    def setup_styles(self):
        self.root.configure(bg="#F3F5F7")
        self.base_font = ("Helvetica", 11)
        self.header_font = ("Helvetica", 12, "bold")
        self.title_font = ("Helvetica", 15, "bold")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#F3F5F7")
        style.configure("Header.TLabel", background="#F3F5F7", font=self.header_font)
        style.configure("Meta.TLabel", background="#F3F5F7", font=self.base_font)
        style.configure("TButton", font=self.base_font, padding=6)
        style.configure("TCombobox", font=self.base_font)
        style.configure("TNotebook", background="#F3F5F7")
        style.configure("TNotebook.Tab", font=self.base_font, padding=(10, 6))

    def setup_ui(self):
        container = ttk.Frame(self.root, padding=14)
        container.pack(fill=tk.BOTH, expand=True)

        title_label = tk.Label(
            container,
            text="Copilot Processed JSON Viewer",
            font=self.title_font,
            bg="#F3F5F7",
            fg="#1F2937",
        )
        title_label.pack(anchor="w", pady=(0, 10))

        controls = ttk.Frame(container)
        controls.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(controls, text="Dataset folder:", style="Header.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.folder_var = tk.StringVar()
        self.folder_combo = ttk.Combobox(
            controls,
            textvariable=self.folder_var,
            state="readonly",
            values=self.dataset_folders,
            width=48,
        )
        self.folder_combo.grid(row=0, column=1, sticky="w")
        self.folder_combo.bind("<<ComboboxSelected>>", lambda _event: self.change_folder())

        ttk.Button(controls, text="Refresh", command=self.refresh_current_folder).grid(row=0, column=2, padx=8)

        ttk.Button(controls, text="Previous", command=self.prev_file).grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Button(controls, text="Next", command=self.next_file).grid(row=1, column=1, sticky="w", pady=(10, 0), padx=(0, 8))

        ttk.Label(controls, text="Jump to file:", style="Header.TLabel").grid(row=1, column=2, sticky="e", pady=(10, 0), padx=(10, 8))
        self.file_var = tk.StringVar()
        self.file_combo = ttk.Combobox(controls, textvariable=self.file_var, state="readonly", width=36)
        self.file_combo.grid(row=1, column=3, sticky="w", pady=(10, 0))
        self.file_combo.bind("<<ComboboxSelected>>", lambda _event: self.jump_to_file())

        meta_frame = ttk.Frame(container)
        meta_frame.pack(fill=tk.X, pady=(0, 10))

        self.folder_label = ttk.Label(meta_frame, text="Folder: -", style="Meta.TLabel")
        self.folder_label.pack(anchor="w")
        self.file_label = ttk.Label(meta_frame, text="File: -", style="Meta.TLabel")
        self.file_label.pack(anchor="w", pady=(2, 0))
        self.index_label = ttk.Label(meta_frame, text="Index: -", style="Meta.TLabel")
        self.index_label.pack(anchor="w", pady=(2, 0))

        notebook = ttk.Notebook(container)
        notebook.pack(fill=tk.BOTH, expand=True)

        field_tab = ttk.Frame(notebook)
        full_tab = ttk.Frame(notebook)
        notebook.add(field_tab, text="Field Browser")
        notebook.add(full_tab, text="Full JSON")

        field_pane = ttk.PanedWindow(field_tab, orient=tk.HORIZONTAL)
        field_pane.pack(fill=tk.BOTH, expand=True)

        field_list_frame = ttk.Frame(field_pane, padding=(0, 0, 10, 0))
        field_pane.add(field_list_frame, weight=1)
        field_value_frame = ttk.Frame(field_pane)
        field_pane.add(field_value_frame, weight=3)

        ttk.Label(field_list_frame, text="Fields", style="Header.TLabel").pack(anchor="w", pady=(0, 6))
        self.field_listbox = tk.Listbox(
            field_list_frame,
            font=self.base_font,
            activestyle="dotbox",
            exportselection=False,
        )
        self.field_listbox.pack(fill=tk.BOTH, expand=True)
        self.field_listbox.bind("<<ListboxSelect>>", lambda _event: self.update_field_value())

        ttk.Label(field_value_frame, text="Field Value", style="Header.TLabel").pack(anchor="w", pady=(0, 6))
        self.field_value_text = scrolledtext.ScrolledText(
            field_value_frame,
            wrap=tk.WORD,
            font=("Courier", 11),
            bg="#FFFFFF",
            fg="#1F2937",
        )
        self.field_value_text.pack(fill=tk.BOTH, expand=True)
        self.field_value_text.configure(state=tk.DISABLED)

        ttk.Label(full_tab, text="Full JSON", style="Header.TLabel").pack(anchor="w", pady=(0, 6))
        self.full_json_text = scrolledtext.ScrolledText(
            full_tab,
            wrap=tk.NONE,
            font=("Courier", 11),
            bg="#FFFFFF",
            fg="#1F2937",
        )
        self.full_json_text.pack(fill=tk.BOTH, expand=True)
        self.full_json_text.configure(state=tk.DISABLED)

        self.root.bind("<Left>", lambda _event: self.prev_file())
        self.root.bind("<Right>", lambda _event: self.next_file())

    def refresh_current_folder(self):
        self.change_folder(preserve_file=True)

    def change_folder(self, preserve_file=False):
        folder_name = self.folder_var.get().strip()
        folder_path = os.path.join(DATA_ROOT, folder_name)

        if not os.path.isdir(folder_path):
            self.current_folder = ""
            self.current_files = []
            self.file_combo["values"] = []
            self.clear_display()
            return

        previous_filename = self.current_files[self.current_index] if preserve_file and self.current_files else None

        self.current_folder = folder_path
        self.current_files = sorted(
            [name for name in os.listdir(folder_path) if name.endswith(".json")],
            key=natural_sort_key,
        )

        self.file_combo["values"] = self.current_files

        if not self.current_files:
            self.current_index = 0
            self.clear_display()
            self.folder_label.config(text=f"Folder: {folder_name}")
            self.file_label.config(text="File: no JSON files found")
            self.index_label.config(text="Index: 0 / 0")
            return

        if previous_filename and previous_filename in self.current_files:
            self.current_index = self.current_files.index(previous_filename)
        else:
            self.current_index = 0

        self.load_current_file()

    def jump_to_file(self):
        selected = self.file_var.get().strip()
        if selected in self.current_files:
            self.current_index = self.current_files.index(selected)
            self.load_current_file()

    def prev_file(self):
        if not self.current_files:
            return
        self.current_index = (self.current_index - 1) % len(self.current_files)
        self.load_current_file()

    def next_file(self):
        if not self.current_files:
            return
        self.current_index = (self.current_index + 1) % len(self.current_files)
        self.load_current_file()

    def load_current_file(self):
        if not self.current_files or not self.current_folder:
            self.clear_display()
            return

        filename = self.current_files[self.current_index]
        file_path = os.path.join(self.current_folder, filename)

        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                self.current_data = json.load(handle)
        except Exception as exc:
            messagebox.showerror("Failed to load JSON", f"Could not load {file_path}\n\n{exc}")
            self.current_data = {}

        self.file_var.set(filename)
        self.update_meta_labels(file_path)
        self.update_field_list()
        self.update_full_json()

    def update_meta_labels(self, file_path):
        folder_name = os.path.basename(self.current_folder) if self.current_folder else "-"
        filename = os.path.basename(file_path)
        self.folder_label.config(text=f"Folder: {folder_name}")
        self.file_label.config(text=f"File: {filename}")
        self.index_label.config(text=f"Index: {self.current_index + 1} / {len(self.current_files)}")

    def update_field_list(self):
        self.field_listbox.delete(0, tk.END)

        keys = sorted(self.current_data.keys(), key=natural_sort_key)
        for key in keys:
            self.field_listbox.insert(tk.END, key)

        if keys:
            self.field_listbox.selection_set(0)
            self.field_listbox.activate(0)
            self.update_field_value()
        else:
            self.write_text(self.field_value_text, "No fields found in this JSON file.")

    def update_field_value(self):
        selection = self.field_listbox.curselection()
        if not selection:
            self.write_text(self.field_value_text, "")
            return

        key = self.field_listbox.get(selection[0])
        value = self.current_data.get(key, "")
        pretty_value = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)
        self.write_text(self.field_value_text, pretty_value)

    def update_full_json(self):
        pretty_json = json.dumps(self.current_data, indent=2, ensure_ascii=False)
        self.write_text(self.full_json_text, pretty_json)

    def clear_display(self):
        self.current_data = {}
        self.field_listbox.delete(0, tk.END)
        self.write_text(self.field_value_text, "")
        self.write_text(self.full_json_text, "")

    @staticmethod
    def write_text(widget, text):
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.configure(state=tk.DISABLED)


def main():
    root = tk.Tk()
    JsonViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()