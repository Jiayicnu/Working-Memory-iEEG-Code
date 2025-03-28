% Support vector machine (SVM) classifier for WM load. 
% Requires LIBSVM package proposed by Chang et al.,2011.
% This classification was performed using power features from entorhinal cortex,hippocampus and lateral temporal cortex,separately.
% The procedure is very similar to decode load 6 vs load 8.
% Input:
    %power_features (struct):z-scored power features:
    %    - zpower_set4_subs: Data for load 4 (num_trials x num_freqs x num_times).
    %    - zpower_set6_subs: Data for load 6 
% Output:
    %   accuracy (vector): Classification accuracy across 100 cross-validation iterations.

function [] = SVM_singleregion(power_features)

    power_set4 = power_features.zpower_set4_subs;
    power_set6 = power_features.zpower_set6_subs;
    power_set4=reshape(power_set4,[size(power_set4,1) size(power_set4,2)*size(power_set4,3)]);
    power_set6=reshape(power_set6,[size(power_set6,1) size(power_set6,2)*size(power_set6,3)]);

    % define number of cross-validation
    num = 100;

 % SVM training and testing
 for i = 1:num
    
    % split all trials' dataset into 70% as training and 30% as testing    
    [train_set4_idx, ~, test_set4_idx] = dividerand(size(power_set4, 1), 0.7, 0, 0.3);
    train_set4 = power_set4(train_set4_idx, :);
    test_set4 = power_set4(test_set4_idx, :);
    [train_set6_idx, ~, test_set6_idx] = dividerand(size(power_set6, 1), 0.7, 0, 0.3);
    train_set6 = power_set6(train_set6_idx, :);
    test_set6 = power_set6(test_set6_idx, :);
    train_label = [ones(size(train_set4, 1), 1); -ones(size(train_set6, 1), 1)];
    test_label = [ones(size(test_set4, 1), 1); -ones(size(test_set6, 1), 1)];
    train = [train_set4; train_set6];
    test = [test_set4; test_set6];
    
   % normalization training and testing dataset
    [Train,PS]=mapminmax(train');
    train_zscore=Train';
    Test=mapminmax('apply',test',PS);
    test_zscore=Test';

    % principle component analysis (PCA)
    [pc,score,latent,tsquare] = pca(train_zscore,'Algorithm','svd','Centered',false);
    tmp=cumsum(latent)./sum(latent);
    indd=find(tmp>=0.99);% keep principle components that explain 99% of the data
    train_pc=train_zscore*pc(:,1:indd(1));           
    test_pc=test_zscore*pc(:,1:indd(1));

    % training and testing model
    parameter=['-s ' num2str(1) ' -t ' num2str(0) ' -c ' num2str(1) ' -g ' num2str(0.1) ' -b 1'];
    model = svmtrain(train_label,train_pc,parameter);             
    [~,acc,dec_values] = svmpredict(test_label,test_pc,model);    
    accuracy(i)=acc(1);
    clear test train pc score latent tsquare tmp indd
 end

end
