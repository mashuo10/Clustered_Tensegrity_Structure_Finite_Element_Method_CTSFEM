function gr = group_numbers_1(N,increment)
    % 输入 N 是数值的范围，例如 40
    % 目标是从 1 到 N 生成类似 (2, 20, 22, 40), (3, 19, 23, 39) 这样的分组
    
    % 初始化一个空的 cell 数组
    gr = cell(N/4+1,1); 
     
    % 特别处理第一组：1和21
    gr{1} = [1+increment, 21+increment];
     % 从第二组开始，每组4个元素，按规律填充
         % 计算前两个数和后两个数的和
    sum1 = N/2 + 2;  % 前两个数的和
    sum2 = N + N/2 + 2;  % 后两个数的和
    for i = 2:N/4
        % 根据规律填充分组
        % 每一组的规律是：i+1 和 N-i+1，N-i+3 和 N-i+19
                % 前两个数的和是 sum1，后两个数的和是 sum2
        % 每一组的两个数会基于这两个和，生成符合规律的数值
        
        % 生成前两个数
        x1 = i;  % 第一个数
        x2 = sum1 - x1;  % 第二个数 (x1 + x2 = sum1)
        
        % 生成后两个数
        x3 = i + 20;  % 第三个数
        x4 = sum2 - x3;  % 第四个数 (x3 + x4 = sum2)
        
        % 将这四个数值加入到当前组中
        gr{i} = [x1+increment, x2+increment, x3+increment, x4+increment];
           end

    % 最后一组的特例处理
    gr{end} = [11+increment, 31+increment];
    
    % 按照每组的第一个元素排序
    [~, sort_idx] = sort(cellfun(@(x) x(1), gr)); 
    gr = gr(sort_idx);  % 根据排序索引排序 gr
end
% 调用函数进行