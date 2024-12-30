function [IMFd,centralFreqs,residuals] = vme(rawSignal,sampleRate,varargin)
%% VME: Variational Mode Extraction
    %
    % 该函数实现了变分模态分解（Variational Mode Extraction, VME），
    % 用于将输入信号分解为所需中心频率的本征模态函数（IMF）
    % 参考文献：Variational Mode Extraction: A New Efficient Method to Derive Respiratory Signals from ECG
    %
    % 输入参数:
    %   rawSignal - 原始信号，通常是一个一维数组或向量。
    %   sampleRate - 采样率，单位为 Hz。信号采样频率。
    %   varargin - 可选参数，以键值对的形式传递。支持以下选项：
    %       'PenaltyFactor' - 惩罚系数 alpha，默认为2000。大小会约束IMF的带宽。
    %       'LMUpdateRate' - 更新率系数 tau，默认为0.01。
    %       'ExpectedIMFd' - 预期的IMF模式，默认为空。暂无实现
    %       'MaxIterations' - 最大迭代次数，默认为500。
    %       'AbsoluteTolerance' - 绝对容差，默认为1e-3。
    %       'CentralFrequencies' - 预设所需分解的IMF的中心频率。
    %
    % 输出参数:
    %   IMFd - 分解得到的中心频率为IMF
    %   centralFreqs - 每个IMF对应的中心频率。
    %   residuals - 残差，表示原始信号与所有IMF之和之间的差异。
    %
    % 示例调用:
    %   [imf, freqs, res] = vme(signal, fs, 'NumIMFd', 4, 'PenaltyFactor', 1500);

%% 初始化
[sgn,info] = setInfo(rawSignal,sampleRate,varargin);

%计算原信号频域 先镜像补偿端点效应 再fft后取半边频谱(实信号频谱对称故取半边)
sgnFDfull = mirrorSignal(sgn,info,'add');%Frequency Domian
sgnFD = sgnFDfull(1:info.halfFreLen);

%初始化模态 初始化为0 并计算频谱(半边)
initIMFd = zeros(info.rawSignLen,1);
initIMFdFDfull = fft(initIMFd,info.fftLen);
initIMFdFD = initIMFdFDfull(1:info.halfFreLen,:);
initIMFdFDNorm = abs(initIMFdFD).^2;

%初始化超参
lambda =  zeros(size(sgnFD));
norFre = ((0:(info.fftLen/2))/info.fftLen).';%归一化频率至[0,0.5](信号最大频率为采样频率一半),表示频域
omega = info.centralFres;%初始化所需提取模态中心频率
iter = 0;
tolerance = inf;

%进入迭代内变量赋值 记录当前与上次迭代结果
IMFdFD = initIMFdFD;
IMFdFDNorm = zeros(size(initIMFdFDNorm));%存放每个模态的平方
lastIMFdFD = IMFdFD;
%% 主循环
while(iter < info.maxiter &&  tolerance > info.diff)

    coeff = (info.alpha^2)*((norFre-omega).^4);
    IMFdFD = (sgnFD + coeff.*lastIMFdFD + lambda/2)./((1 + coeff).*(1 + 2*info.alpha*((norFre-omega).^2)));
    IMFdFDNorm = abs(IMFdFD).^2;
    omega = norFre.'*IMFdFDNorm./sum(IMFdFDNorm);
    lambda = lambda + info.tau*((sgnFD - IMFdFD)./(1 + coeff));

    tolerance = sum(abs(IMFdFD - lastIMFdFD).^2)/sum(abs(lastIMFdFD).^2);

    lastIMFdFD = IMFdFD;
    iter = iter + 1;
end
%% 输出结果
if (tolerance > info.diff)
    warning('VMD failed to converge!');%分解失败
end
IMFdFDfull = complex(zeros(info.fftLen,1));%对分解完的模态进行扩展 镜像后转回时域
IMFdFDfull(1:info.halfFreLen,:) = IMFdFD;
IMFdFDfull(info.halfFreLen+1:end,:) = conj(IMFdFD(end-1:-1:2,:));%实信号做fft后共轭对称

IMFd = mirrorSignal(IMFdFDfull,info,'remove');
centralFreqs = omega*sampleRate;
residuals = rawSignal - IMFd;
residuals = tolerance;
end

%% 镜像信号函数 减少端点效应
%  将原信号拆成两半 前一半翻转补充在原信号前面 后一半翻转补充在原信号后面
function y = mirrorSignal(x,info,operate)
    if strcmp(operate, 'remove')
        xr = real(ifft(x));
        y = xr(info.halfrawSignLen+1:info.mirSignLen-info.halfrawSignLen,:);
    elseif strcmp(operate, 'add')
        xm = [x(info.halfrawSignLen:-1:1); x; x(info.rawSignLen:-1:info.halfrawSignLen+1)];
        y = fft(xm);
    end
end

%% 默认参数设置
function [sgn,info] = setInfo(rawSignal,sampleRate,varargin)
    if (mod(length(rawSignal), 2) == 1)
        rawSignal = [rawSignal rawSignal(end)];
    end
    sgn = rawSignal(:);
    info.alpha = 2000;
    info.tau = 0.01;
    info.maxiter = 500;
    info.expectedIMFd = [];
    info.diff = 1e-3;
    for i = 1:2:length(varargin{1})-1 %vsd中传入的varargin为1*n的cell，将varargin再次作为变量传入为1*1的cell，里面包含{1*n的cell}
        varKey = varargin{1}{i};
        varVal = varargin{1}{i+1};
        switch varKey
            case 'PenaltyFactor'
                info.alpha = varVal;
            case 'LMUpdateRate'
                info.tau = varVal;
            case 'ExpectedIMFd' %预期模态
                info.expectedIMFd = varVal;%列向量
            case 'MaxIterations'
                info.maxiter = varVal;
            case 'AbsoluteTolerance'
                info.diff = varVal;
            case 'CentralFrequencies'
                info.centralFres = varVal/sampleRate;
            otherwise
                error('Invalid parameter name');
        end
    end
    info.rawSignLen = length(sgn);
    info.halfrawSignLen = length(sgn)/2;%镜像分割信号长度
    info.mirSignLen = info.rawSignLen + info.halfrawSignLen*2;%镜像信号总长度
    info.fftLen = info.mirSignLen;
    info.halfFreLen = info.fftLen/2 + 1;
end
    