import subprocess

# Path to the application executable
app_path = r"C:\SoulSaverOnline\SoulSaver.exe"

# Number of instances you want to open
num_instances = 5

# Launch multiple instances
for _ in range(num_instances):
    subprocess.Popen([app_path])
