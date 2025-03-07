from moviepy.editor import VideoFileClip

def convert_mp4_to_gif(input_path, output_path, start_time=0, end_time=None, fps=10):
    clip = VideoFileClip(input_path).subclip(start_time, end_time)
    clip.write_gif(output_path, fps=fps)

# Usage
convert_mp4_to_gif('ora2.mp4', 'ora2.gif')