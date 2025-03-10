%
%
%
clear;
clc;
% data_path = fullfile(pwd, '..',  filesep, "data_mv", filesep);
data_path = fullfile(pwd,  "data");
addpath(data_path);
lib_path = fullfile(pwd, "lib", filesep);
addpath(lib_path);
code_path = fullfile(pwd, "ours", filesep);
addpath(genpath(code_path));


dirop = dir(fullfile(data_path, '*.mat'));
datasetCandi = {dirop.name};

exp_n = 'DMOE_MVKM_v2_results';

for i1 = 1:length(datasetCandi)
    data_name = datasetCandi{i1}(1:end-4);
    dir_name = [pwd, filesep, exp_n, filesep, data_name];
    create_dir(dir_name);
    
    clear Xs Y;
    load(data_name);
    
    if size(Xs, 1) > 1 && size(Xs, 2) == 1
        Xs = Xs';
    end
    
    nView = length(Xs);
    nSmp = length(Y);
    nCluster = length(unique(Y));
    nGroup_candidate = [1:10];
    
    %*********************************************************************
    % DMOE_MVKM_v2
    %*********************************************************************
    nRepeat = 10;
    nMeasure = 7;
    nParam = length(nGroup_candidate);
    DMOE_MVKM_v2_res = zeros(nParam, nRepeat, nMeasure);
    DMOE_MVKM_v2_res_his = cell(nParam, nRepeat);
    DMOE_MVKM_v2_time = zeros(nParam, nRepeat);
    fname2 = fullfile(dir_name, [data_name, '_', exp_n, '_res.mat']);
    seedplus = 0;
    if ~exist(fname2, 'file')
        
        for iParam = 1:nParam
            disp([exp_n, ' iParam= ', num2str(iParam), ', totalParam= ', num2str(nParam)]);
            nGroup = nGroup_candidate(iParam);
            for iRepeat = 1:nRepeat
                disp([exp_n,  ' ', data_name, '        ', num2str(iRepeat), '/', num2str(nRepeat)]);
                fname3 = fullfile(dir_name, [data_name, '_', exp_n, '_param', num2str(iParam), '_repeat', num2str(iRepeat), '.mat']);
                if exist(fname3, 'file')
                    clear res_i t
                    load(fname3, 'res_i', 't');
                else
                    t1_s = tic;
                    rand('twister', 5489+seedplus);
                    [label, Yc, Z, Gs, objHistory] = DMOE_MVKM_v2(Xs, nCluster, nGroup);
                    t = toc(t1_s);
                    res_i = my_eval_y_2025_m7(label, Y)';
                    DMOE_MVKM_v2_res(iParam, iRepeat, :) = res_i;
                    save(fname3, 'res_i', 't');
                end
                seedplus = seedplus + 1;
                DMOE_MVKM_v2_time(iParam, iRepeat) = t;
                DMOE_MVKM_v2_res(iParam, iRepeat, :) = res_i;
            end
        end
        DMOE_MVKM_v2_res_mean_param = reshape(mean(DMOE_MVKM_v2_res, 2), size(DMOE_MVKM_v2_res, 1), size(DMOE_MVKM_v2_res, 3));
        DMOE_MVKM_v2_res_std_param = zeros(size(DMOE_MVKM_v2_res, 1), size(DMOE_MVKM_v2_res, 3));
        for iParam = 1:size(DMOE_MVKM_v2_res, 1)
            tmp = reshape(DMOE_MVKM_v2_res(iParam,:,:), size(DMOE_MVKM_v2_res, 2), size(DMOE_MVKM_v2_res, 3));
            DMOE_MVKM_v2_res_std_param(iParam, :) = std(tmp);
        end
        [DMOE_MVKM_v2_res_good, DMOE_MVKM_v2_res_good_idx] = max(DMOE_MVKM_v2_res_mean_param, [], 1);
        DMOE_MVKM_v2_result_summary = [DMOE_MVKM_v2_res_good,  mean(DMOE_MVKM_v2_time)];
        param_lidx = sub2ind(size(DMOE_MVKM_v2_res_std_param), DMOE_MVKM_v2_res_good_idx, (1:size(DMOE_MVKM_v2_res_std_param, 2)));
        DMOE_MVKM_v2_std_summary = [DMOE_MVKM_v2_res_std_param(param_lidx), mean(DMOE_MVKM_v2_time(:))] ;
        save(fname2, 'DMOE_MVKM_v2_result_summary','DMOE_MVKM_v2_std_summary', 'DMOE_MVKM_v2_res', 'DMOE_MVKM_v2_time','DMOE_MVKM_v2_res_good_idx');
    end
    disp([data_name, ' has been completed!']);
end

rmpath(data_path);
rmpath(lib_path);
rmpath(code_path);
clear; clc;