# Deep Learning Approach to Global Bird's Eye View Semantic Mapping

# Setup
1. Clone the repo
2. Create `results` and `temp` folder under `ai_imu_dr`
3. Download the dataset from [ai-imu's dropbox](https://www.dropbox.com/s/ey41xsvfqca30vv/data.zip)
4. Download the local map data from [Google Drive](https://drive.google.com/file/d/1i5vMMN_DKbk3g-sNs-eEluKYWHowTE_-/view?usp=share_link)
5. Create `temp` and `results` folder under root directory.
6. Unzip the data under the root directory, the folder name should be `can_bus`.
7. Set the working directory to root.
8. The directory should look like this: ![directory](readme.media/readme.directory.png)
9. Run `python3 kitti_mapping.py`
10. If you want to run `main_KITTI.py`, please call it under root directory, and run `python3 ai_imu_dr/src/main_kitti.py`

# Things need to be implemented
## Map local sematic label to global
- [ ] Get grid size from MotionNet
- [ ] Requires coordinate transformation, convert BEV local cells to global cell coordinate.
## Mapping 
- Not able to start yet
- [ ] Loop over poses and convert local maps to global grid.
- [ ] Update map parameters for each cell (discrete sematic counting sensor model)