"""
Video Frame Extractor for Parking Slot Alignment
This tool extracts frames from your video to help create properly aligned parking slots
"""

import cv2
import os
import sys

class VideoFrameExtractor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video Properties:")
        print(f"- Resolution: {self.width}x{self.height}")
        print(f"- FPS: {self.fps:.2f}")
        print(f"- Total Frames: {self.frame_count}")
        print(f"- Duration: {self.duration:.2f} seconds")
        print("-" * 40)

    def extract_frame_by_number(self, frame_number, output_name="reference_frame.jpg"):
        """Extract a specific frame by frame number"""
        if frame_number >= self.frame_count or frame_number < 0:
            print(f"Error: Frame number {frame_number} is out of range (0-{self.frame_count-1})")
            return False

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        if ret:
            cv2.imwrite(output_name, frame)
            timestamp = frame_number / self.fps if self.fps > 0 else 0
            print(f"✓ Extracted frame {frame_number} (at {timestamp:.2f}s) -> {output_name}")
            return True
        else:
            print(f"Error: Could not read frame {frame_number}")
            return False

    def extract_frame_by_time(self, seconds, output_name="reference_frame.jpg"):
        """Extract a frame at specific time in seconds"""
        frame_number = int(seconds * self.fps)
        return self.extract_frame_by_number(frame_number, output_name)

    def extract_multiple_frames(self, frame_numbers, output_prefix="frame"):
        """Extract multiple frames"""
        success_count = 0
        for i, frame_num in enumerate(frame_numbers):
            output_name = f"{output_prefix}_{frame_num:06d}.jpg"
            if self.extract_frame_by_number(frame_num, output_name):
                success_count += 1

        print(f"\nExtracted {success_count}/{len(frame_numbers)} frames successfully")
        return success_count

    def interactive_extractor(self):
        """Interactive frame extraction with preview"""
        print("\nInteractive Frame Extractor")
        print("Controls:")
        print("- Press SPACE to extract current frame")
        print("- Press 'q' to quit")
        print("- Press 'n' to jump to next 100 frames")
        print("- Press 'p' to jump to previous 100 frames")
        print("- Press 'f' to go to first frame")
        print("- Press 'l' to go to last frame")
        print("- Press 's' to enter specific frame number")
        print("-" * 40)

        current_frame = 0
        extracted_count = 0

        while True:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = self.cap.read()

            if not ret:
                print("End of video reached")
                break

            # Add frame info overlay
            info_text = f"Frame: {current_frame}/{self.frame_count-1} | Time: {current_frame/self.fps:.2f}s"
            cv2.rectangle(frame, (10, 10), (600, 50), (0, 0, 0), -1)
            cv2.putText(frame, info_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display frame
            cv2.namedWindow("Video Frame Extractor", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Video Frame Extractor", 1280, 720)
            cv2.imshow("Video Frame Extractor", frame)

            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):  # Space to extract
                output_name = f"extracted_frame_{current_frame:06d}.jpg"
                cv2.imwrite(output_name, frame)
                extracted_count += 1
                print(f"✓ Extracted frame {current_frame} -> {output_name}")
            elif key == ord('n'):  # Next 100 frames
                current_frame = min(current_frame + 100, self.frame_count - 1)
            elif key == ord('p'):  # Previous 100 frames
                current_frame = max(current_frame - 100, 0)
            elif key == ord('f'):  # First frame
                current_frame = 0
            elif key == ord('l'):  # Last frame
                current_frame = self.frame_count - 1
            elif key == ord('s'):  # Specific frame
                try:
                    frame_input = input("\nEnter frame number (0 to {}): ".format(self.frame_count-1))
                    new_frame = int(frame_input)
                    if 0 <= new_frame < self.frame_count:
                        current_frame = new_frame
                    else:
                        print("Invalid frame number!")
                except (ValueError, EOFError):
                    print("Invalid input!")

        cv2.destroyAllWindows()
        print(f"\nSession complete. Extracted {extracted_count} frames total.")

    def extract_evenly_spaced_frames(self, num_frames=10, output_prefix="sample"):
        """Extract evenly spaced frames throughout the video"""
        if num_frames > self.frame_count:
            num_frames = self.frame_count

        step = max(1, self.frame_count // num_frames)
        frame_numbers = [i * step for i in range(num_frames)]

        print(f"Extracting {num_frames} evenly spaced frames...")
        return self.extract_multiple_frames(frame_numbers, output_prefix)

    def close(self):
        """Close the video capture"""
        if self.cap:
            self.cap.release()


def main():
    if len(sys.argv) < 2:
        print("Video Frame Extractor for Parking Slot Alignment")
        print("=" * 50)
        print("Usage: python img.py <video_path> [options]")
        print()
        print("Options:")
        print("  -i, --interactive    Launch interactive extractor")
        print("  -f, --frame <num>    Extract specific frame number")
        print("  -t, --time <sec>     Extract frame at specific time")
        print("  -s, --sample <num>   Extract evenly spaced frames")
        print("  -o, --output <name>  Output filename (default: reference_frame.jpg)")
        print()
        print("Examples:")
        print("  python img.py video.mp4 -i")
        print("  python img.py video.mp4 -f 1500")
        print("  python img.py video.mp4 -t 30.5")
        print("  python img.py video.mp4 -s 5")
        print()
        return

    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        return

    try:
        extractor = VideoFrameExtractor(video_path)

        # Parse command line arguments
        args = sys.argv[2:]

        if not args or '-i' in args or '--interactive' in args:
            # Interactive mode
            extractor.interactive_extractor()

        elif '-f' in args or '--frame' in args:
            # Extract specific frame
            try:
                idx = args.index('-f') if '-f' in args else args.index('--frame')
                frame_num = int(args[idx + 1])

                output_name = "reference_frame.jpg"
                if '-o' in args:
                    output_idx = args.index('-o')
                    output_name = args[output_idx + 1]
                elif '--output' in args:
                    output_idx = args.index('--output')
                    output_name = args[output_idx + 1]

                extractor.extract_frame_by_number(frame_num, output_name)

            except (IndexError, ValueError):
                print("Error: Invalid frame number specified")

        elif '-t' in args or '--time' in args:
            # Extract frame at specific time
            try:
                idx = args.index('-t') if '-t' in args else args.index('--time')
                time_sec = float(args[idx + 1])

                output_name = "reference_frame.jpg"
                if '-o' in args:
                    output_idx = args.index('-o')
                    output_name = args[output_idx + 1]
                elif '--output' in args:
                    output_idx = args.index('--output')
                    output_name = args[output_idx + 1]

                extractor.extract_frame_by_time(time_sec, output_name)

            except (IndexError, ValueError):
                print("Error: Invalid time specified")

        elif '-s' in args or '--sample' in args:
            # Extract evenly spaced frames
            try:
                idx = args.index('-s') if '-s' in args else args.index('--sample')
                num_frames = int(args[idx + 1])

                output_prefix = "sample_frame"
                if '-o' in args:
                    output_idx = args.index('-o')
                    output_prefix = args[output_idx + 1]
                elif '--output' in args:
                    output_idx = args.index('--output')
                    output_prefix = args[output_idx + 1]

                extractor.extract_evenly_spaced_frames(num_frames, output_prefix)

            except (IndexError, ValueError):
                print("Error: Invalid number of frames specified")
        else:
            print("No valid option specified. Use -h for help or run without arguments for interactive mode.")

    except Exception as e:
        print(f"Error: {str(e)}")

    finally:
        try:
            extractor.close()
        except:
            pass


if __name__ == "__main__":
    main()