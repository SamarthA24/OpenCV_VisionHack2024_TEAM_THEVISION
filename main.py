import cv2
import numpy as np
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import matplotlib.pyplot as plt

# Add your heatmap generation function
def generate_player_heatmap(tracks, player_id, field_image_path="output_videos/screenshot.png"):
    """
    Generate a heatmap for a player based on their positions over time.
    """
    # Load the field image for background
    field_img = cv2.imread(field_image_path)
    
    # Ensure the field image is the right size
    height, width, _ = field_img.shape

    # Initialize a blank heatmap
    heatmap = np.zeros((height, width), dtype=np.float32)

    # Loop through the player tracks and accumulate position data
    for frame_num, player_track in enumerate(tracks['players']):
        if player_id in player_track:
            player_data = player_track[player_id]
            if 'position' in player_data:  # Ensure position data is available
                position = player_data['position']
                x, y = int(position[0]), int(position[1])
                if 0 <= x < width and 0 <= y < height:
                    heatmap[y, x] += 1  # Increment heatmap at the player's position

    # Normalize the heatmap
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the heatmap to a color image
    heatmap_colored = cv2.applyColorMap(np.uint8(heatmap), cv2.COLORMAP_JET)

    # Combine the heatmap with the field image (for visualization)
    overlay = cv2.addWeighted(field_img, 0.7, heatmap_colored, 0.3, 0)

    # Save the heatmap as a .png file
    heatmap_image_path = f"player_{player_id}_heatmap.png"
    cv2.imwrite(heatmap_image_path, overlay)

    # Display the heatmap (optional)
    plt.imshow(overlay)
    plt.axis('off')  # Hide axes
    plt.show()

    return heatmap_image_path  # Return the file path for reference

def main():
    # Read Video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    # Initialize Tracker
    tracker = Tracker('yolov8n.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    # Get object positions
    tracker.add_position_to_tracks(tracks)

    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Transformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Assign Ball Acquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # Draw output
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

    # Generate and display/save the heatmap for a specific player (e.g., player 1)
    heatmap_image_path = generate_player_heatmap(tracks, player_id=1, field_image_path="output_videos/screenshot.png")
    print(f"Heatmap saved at: {heatmap_image_path}")

if __name__ == '__main__':
    main()
