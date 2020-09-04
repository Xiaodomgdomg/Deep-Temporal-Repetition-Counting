

clear all;

root = '';
load_root = fullfile(root, 'repetition_label_c01only');
save_root = fullfile(root, 'repetition_label_c01only_interpolation');

dir_path = dir(load_root);

dataset_fail_num = 0;
dataset_num = 0;
dataset_cnt = 0;

ptr=1;
for i = 11:numel(dir_path)
    
    fid_read = fopen(fullfile(load_root, [dir_path(i).name]), 'r');
    
    
    for j = 1:25
        
        file_name = fscanf(fid_read, '%s ', 1);  
        tmp = fscanf(fid_read, '%d');
        
        cnt_num = numel(tmp);
        if cnt_num == 1
            dataset_fail_num = dataset_fail_num + 1;
        else
            tmp2 = tmp(2:end)-tmp(1:end-1);
            
            % estimation of dataset distribution
            dataset_num = dataset_num + 1;
            dataset_cnt = dataset_cnt + numel(tmp);
            dataset.duration(ptr) = tmp(end)-tmp(1)+1;
            dataset.count(ptr) = numel(tmp);
            dataset.length_variation(ptr) = (max(tmp2)-min(tmp2))/mean(tmp2);
            
            clear label;
            
            label.duration = dataset.duration(ptr);
            label.start_frame = tmp(1);
            label.end_frame = tmp(end);
            label.temporal_bound_num = dataset.count(ptr);
            label.temporal_bound = tmp;
            
            for k = 1:numel(tmp)-1
                for p = tmp(k):tmp(k+1)
                    seg_m = tmp(k+1)-tmp(k);
                    if seg_m <= 0
                        raise('error');
                    end
                    seg_l = -1;
                    seg_r = -1;
                    
                    if k == numel(tmp)-1
                        label.offset_next_estimate(p-tmp(1)+1) = -1;
                    else
                        seg_r = tmp(k+2)-tmp(k+1);
                        if max(seg_m, seg_r) - min(seg_m, seg_r) > 2 && max(seg_m, seg_r) / min(seg_m, seg_r) > 1.3
                            label.offset_next_estimate(p-tmp(1)+1) = -1;
                        else
                            offset_next = (p-tmp(k)) / seg_m * seg_r + tmp(k+1) - p;
                            offset_next = int32(round(offset_next));
                            label.offset_next_estimate(p-tmp(1)+1) = offset_next;
                        end
                    end
                        
                    if k == 1
                        label.offset_pre_estimate(p-tmp(1)+1) = -1;
                    else
                        seg_l = tmp(k)-tmp(k-1);
                        if max(seg_m, seg_l) - min(seg_m, seg_l) > 2 && max(seg_m, seg_l) / min(seg_m, seg_l) > 1.3
                            label.offset_pre_estimate(p-tmp(1)+1) = -1;
                        else
                            offset_next = p - tmp(k-1) - (p-tmp(k)) / seg_m * seg_l;
                            offset_next = int32(round(offset_next));
                            label.offset_pre_estimate(p-tmp(1)+1) = offset_next;
                        end
                    end
                    
                end
            end
            
            % fullfile(save_root, [file_name '.mat'])
            savepath = fullfile(save_root, 'mat', [file_name '.mat']);
            save(savepath, 'label');
            % save savepath
            data{i,j} = label;
            ptr = ptr+1;
            
        end
    end
    fclose(fid_read);
    
end
