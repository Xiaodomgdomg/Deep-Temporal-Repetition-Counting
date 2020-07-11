files = dir('data/ori_data/YT_seg/videos');
save_path = 'data/ori_data/YT_seg/imgs/val';

for i = 3:numel(files)
    i
    obj = VideoReader([files(i).folder '/' files(i).name]);
    obj_numberofframe = obj.NumberOfFrame;
    
    mkdir([save_path '/' files(i).name(1:end-4)]);
    for j = 1:obj_numberofframe
        %imgs = read(obj,j);
        % imwrite(imgs, [save_path '/' files(i).name(1:end-4) '/' num2str(j,'%06d') '.jpg']);
    end
    fd_frames = fopen([save_path '/' files(i).name(1:end-4) '/' 'n_frames'], 'w');
    fprintf(fd_frames, '%d\n', obj_numberofframe);
    fclose(fd_frames);

    ytseg.duration(i-2) = obj.Duration;
    
    clear obj frame
end
