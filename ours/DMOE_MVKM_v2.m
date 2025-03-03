function  [label, Yc, Z, Gs, objHistory] = DMOE_MVKM_v2(Xs, nCluster, nGroup)
%************************************************
% Xs  1*v cell, n*d^r
% Y discrete  n*v
% Z k*v sum equals to 1 on column
% Gs 1*v cell, discrete n*m
% Us v*m cell, each cell is c*d^r
%************************************************

nView = length(Xs);
nSmp = size(Xs{1}, 1);

%************************************************
% Initalize
%************************************************
Gs = cell(1,nView);
Us = cell(nView, nGroup);
Xn = zeros(nSmp, nView);
nGroups_real = zeros(1, nView);
for iView = 1 : nView
    Xi = Xs{iView};
    Xn(:, iView) = sum(Xi.^2, 2);
    label_initgamma = litekmeans(Xi, nGroup, 'MaxIter', 10, 'Replicates', 1);
    assert(length(unique(label_initgamma)) == nGroup);
    if min(sum(ind2vec(label_initgamma'), 2)) < 5 * nCluster % avoid small group
        label_initgamma = randi(nGroup, nSmp, 1);
    end
    Gi = ind2vec(label_initgamma')' > 0;
    nGroups_real(iView) = size(Gi, 2);
    for iGroup = 1 : nGroups_real(iView)
        [~ , Us{iView, iGroup}] = litekmeans(Xi(Gi(:, iGroup), :), nCluster, 'MaxIter', 10, 'Replicates', 10);
    end
    Gs{iView} = Gi;
end
Z = ones(nCluster, nView);
Z = bsxfun(@rdivide, Z, sum(Z, 2));
hs = Z.^(2);


[Xc, ~, ~] = mySVD(cell2mat(Xs), nCluster);
label = litekmeans(NormalizeFea(Xc), nCluster, 'MaxIter', 100, 'Replicates', 10);
Yc = ind2vec(label')';
lidx = sub2ind([nSmp, nCluster], (1:nSmp)', label);

objHistory = [];
converges = false;
maxIter = 50;
iter = 0;
myeps = 1e-5;

while ~converges
    %************************************************
    % Update Z
    %************************************************

    Z = zeros(nCluster, nView);
    for iView = 1:nView
        Xi = Xs{iView};
        Gi = Gs{iView};
        for iCluster = 1:nCluster
            Gi_c = bsxfun(@and, Gi, Yc(:, iCluster));
            ei_c_g = zeros(nGroups_real(iView), 1);
            for iGroup = 1:nGroups_real(iView)
                idx = Gi_c(:, iGroup);
                Xi_c_g = Xi(idx, :);
                u_c_g = Us{iView, iGroup}(iCluster, :);
                Xi_n = Xn(idx, iView);
                u_c_g_n = sum(u_c_g.^2);
                ei_c_g_0 = Xi_n - 2 * (Xi_c_g *  u_c_g') + u_c_g_n;
                ei_c_g(iGroup) = sum(ei_c_g_0);
            end
            Z(iCluster, iView) = sum(ei_c_g);
        end
    end
    Z = Z.^(-1);   %%%
    Z = bsxfun(@rdivide, Z, sum(Z, 2));
    hs = Z.^(2);  %%%

    %************************************************
    % Update Us
    %************************************************
    for iView = 1 : nView
        Xi = Xs{iView};
        Gi = Gs{iView};
        for iGroup = 1 : nGroups_real(iView)
            for iCluster = 1 : nCluster
                idx = Yc(:, iCluster) .* Gi(:, iGroup);
                idx = idx > 0;
                Us{iView, iGroup}(iCluster, :) = mean(Xi(idx, :), 1);
            end
        end
    end

    %************************************************
    % Update Gs
    %************************************************

    for iView = 1:nView
        Xi = Xs{iView};
        es_g = zeros(nSmp, nGroups_real(iView));
        hi = hs(:, iView);
        for iGroup = 1:nGroups_real(iView)
            Ui_g = Us{iView, iGroup};
            Ui_g_n = sum(Ui_g.^2, 2);
            Ei_g_0 = Xi * Ui_g';
            Ei_g = Ei_g_0(lidx);
            ei_g = Xn(:, iView) - 2 * Ei_g + Ui_g_n(label);
            es_g(:, iGroup) = ei_g .* hi(label);
        end
        [~, label_g] = min(es_g, [], 2);
        Gs{iView} = ind2vec(label_g')' > 0;
        nGroups_real(iView) = size(Gs{iView}, 2);
        if nGroup > size(Gs{iView}, 2)
            disp(' Empty group when update G');
        end
    end

    %************************************************
    % Update Y
    %************************************************

    Es = cell(1, nView); % memory cost
    for iView = 1 : nView
        Xi = Xs{iView};
        Gi = Gs{iView};
        hi = hs(:, iView);
        es_c = zeros(nSmp, nCluster);
        for iGroup = 1:nGroups_real(iView)
            gIdx = Gi(:, iGroup);
            Xi_g = Xi(gIdx, :);
            Ui_g = Us{iView, iGroup};
            Ui_g_n = sum(Ui_g.^2, 2);
            Ei_g_0 = Xi_g * Ui_g';
            Dist2 = bsxfun(@plus, Xn(gIdx, iView), Ui_g_n') - 2 * Ei_g_0;
            wDist2 = bsxfun(@times, Dist2, hi');
            es_c(gIdx, :) = wDist2;
        end
        Es{iView} = es_c;
    end
    es_c = zeros(nSmp, nCluster);
    for iView = 1 : nView
        es_c = es_c + Es{iView};
    end
    [~, label] = min(es_c, [], 2);
    Yc = ind2vec(label')';
    lidx = sub2ind([nSmp, nCluster], (1:nSmp)', label);
    
    if nCluster > size(Yc, 2)
        disp(' Empty cluster when update Y');
    end
 
    obj_fast = sum(es_c(lidx));
    objHistory = [objHistory; obj_fast]; %#ok
    
    iter = iter + 1;
        if iter > 1 && (abs( (objHistory(end-1) - objHistory(end))/objHistory(end-1) )< myeps) || iter > maxIter
            converges = true;
        end
    
end
end