from PIL import Image


img = Image.open("Traj_Velocity.png")
img = img.resize((img.width//5, img.height//5), Image.LANCZOS)  # Downscale by 50%
img.save("Traj_Velocity_c.png", quality=100)