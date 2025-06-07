
import tkinter as tk
from tkinter import messagebox
from simulation import run_simulation

MAX_DRONES = 6
entries = []
defaults = [
    ("0,0,0", "10,10,10", "1.0"),
    ("10,0,0", "0,10,10", "1.0"),
    ("0,10,0", "10,0,10", "1.0")
]

def get_input_and_run():
    try:
        drones = []
        for i, (start_entry, end_entry, speed_entry) in enumerate(entries):
            start = list(map(float, start_entry.get().split(',')))
            end = list(map(float, end_entry.get().split(',')))
            speed = float(speed_entry.get())
            drones.append({
                'name': f'Drone {chr(65+i)}',
                'start': start,
                'end': end,
                'speed': speed
            })
        root.destroy()
        run_simulation(drones)
    except Exception as e:
        messagebox.showerror("Input Error", str(e))

def add_drone():
    i = len(entries)
    if i >= MAX_DRONES:
        messagebox.showinfo("Limit Reached", "Maximum 6 drones allowed.")
        return

    frame = tk.LabelFrame(root, text=f"Drone {chr(65+i)}")
    frame.pack(padx=10, pady=5, fill="x")

    tk.Label(frame, text="Start (x,y,z):").grid(row=0, column=0)
    start_entry = tk.Entry(frame)
    start_entry.insert(0, "0,0,0")
    start_entry.grid(row=0, column=1)

    tk.Label(frame, text="End (x,y,z):").grid(row=1, column=0)
    end_entry = tk.Entry(frame)
    end_entry.insert(0, "10,10,10")
    end_entry.grid(row=1, column=1)

    tk.Label(frame, text="Speed:").grid(row=2, column=0)
    speed_entry = tk.Entry(frame)
    speed_entry.insert(0, "1.0")
    speed_entry.grid(row=2, column=1)

    entries.append((start_entry, end_entry, speed_entry))

root = tk.Tk()
root.title("Drone Collision Simulation Setup")

# 默认添加3个无人机输入
for i in range(3):
    frame = tk.LabelFrame(root, text=f"Drone {chr(65+i)}")
    frame.pack(padx=10, pady=5, fill="x")

    tk.Label(frame, text="Start (x,y,z):").grid(row=0, column=0)
    start_entry = tk.Entry(frame)
    start_entry.insert(0, defaults[i][0])
    start_entry.grid(row=0, column=1)

    tk.Label(frame, text="End (x,y,z):").grid(row=1, column=0)
    end_entry = tk.Entry(frame)
    end_entry.insert(0, defaults[i][1])
    end_entry.grid(row=1, column=1)

    tk.Label(frame, text="Speed:").grid(row=2, column=0)
    speed_entry = tk.Entry(frame)
    speed_entry.insert(0, defaults[i][2])
    speed_entry.grid(row=2, column=1)

    entries.append((start_entry, end_entry, speed_entry))

tk.Button(root, text="➕ Add Drone", command=add_drone).pack(pady=5)
tk.Button(root, text="🚀 Start Simulation", command=get_input_and_run).pack(pady=10)

root.mainloop()
