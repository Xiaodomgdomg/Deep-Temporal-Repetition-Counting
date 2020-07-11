files = dir('data/ori_data/QUVA/videos');
save_path = 'data/ori_data/QUVA/imgs/val';

for i = 3:numel(files
    i
    obj = VideoReader([files(i).folder '/' files(i).name]);
    obj_numberofframe = obj.NumberOfFrame;
    mkdir([save_path '/' files(i).name(1:end-4)]);
    for j = 1:obj_numberofframe
        imgs = read(obj,j);
        imwrite(imgs, [save_path '/' files(i).name(1:end-4) '/' num2str(j,'%06d') '.jpg']);
    end
    clear obj frame
end
