"""
Script to initialize and populate the media assets library.
Run this once to create the media_library.json file with all your assets.
"""

from utils.media_assets_manager import MediaAssetsLibrary

# Initialize the library with your media base path
media_library = MediaAssetsLibrary(base_media_path=".")

# ============================================================================
# ADD TEACH PENDANT IMAGES (images_cobot folder)
# ============================================================================
# Format: media_library.add_image(filename, category, description)

media_library.add_image(
    "teach_pendant_power_on.png",
    "teach_pendant",
    "Power button location on the teach pendant"
)

media_library.add_image(
    "teach_pendant_home_position.png",
    "teach_pendant",
    "Setting the home position using teach pendant controls"
)

media_library.add_image(
    "teach_pendant_installation_file.png",
    "teach_pendant",
    "Show how to create installation file using teach pendant controls"
)

media_library.add_image(
    "teach_pendant_joint_movement.png",
    "teach_pendant",
    "Moving individual robot joints with the teach pendant"
)

# Add more teach pendant images here...
# media_library.add_image("...", "teach_pendant", "...")


# ============================================================================
# ADD COBOT ACTION IMAGES (images_teach_pendant folder)
# ============================================================================
# Format: media_library.add_image(filename, category, description)

media_library.add_image(
    "cobot_screwing_motion.png",
    "cobot_action",
    "Cobot performing screwing motion with screwdriver tool"
)

media_library.add_image(
    "cobot_gripper_attachment.png",
    "cobot_action",
    "Screwdriver gripper properly attached to cobot wrist"
)

media_library.add_image(
    "cobot_approach_position.png",
    "cobot_action",
    "Cobot approaching the assembly workpiece"
)

# Add more cobot action images here...
# media_library.add_image("...", "cobot_action", "...")


# ============================================================================
# ADD VIDEOS (videos folder)
# ============================================================================
# Format: media_library.add_video(filename, description)

media_library.add_video(
    "cobot_screwdriving_demo.mp4",
    "Complete demonstration of the cobot screwdriving process"
)

media_library.add_video(
    "teach_pendant_basic_controls.mp4",
    "Tutorial on basic teach pendant controls and navigation"
)

media_library.add_video(
    "assembly_workflow_overview.mp4",
    "Overview of the complete assembly workflow and steps"
)

# Add more videos here...
# media_library.add_video("...", "...")


# ============================================================================
# SAVE THE LIBRARY
# ============================================================================

# Save the complete media library to JSON
output_path = "settings/media_library.json"
if media_library.save_to_json(output_path):
    print(f"✓ Media library successfully created at {output_path}")
    print(f"✓ Total assets: {len(media_library.get_all_assets())}")
    print(f"  - Images (teach pendant): {len(media_library.get_assets_by_category('teach_pendant'))}")
    print(f"  - Images (cobot action): {len(media_library.get_assets_by_category('cobot_action'))}")
    print(f"  - Videos: {len([a for a in media_library.get_all_assets() if a.asset_type == 'video'])}")
else:
    print("✗ Failed to create media library")