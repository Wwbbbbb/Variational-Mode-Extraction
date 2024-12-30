function [IMFd,centralFreqs,residuals] = vme(rawSignal,sampleRate,varargin)
%% VME: Variational Mode Extraction
    %
    % �ú���ʵ���˱��ģ̬�ֽ⣨Variational Mode Extraction, VME����
    % ���ڽ������źŷֽ�Ϊ��������Ƶ�ʵı���ģ̬������IMF��
    % �ο����ף�Variational Mode Extraction: A New Efficient Method to Derive Respiratory Signals from ECG
    %
    % �������:
    %   rawSignal - ԭʼ�źţ�ͨ����һ��һά�����������
    %   sampleRate - �����ʣ���λΪ Hz���źŲ���Ƶ�ʡ�
    %   varargin - ��ѡ�������Լ�ֵ�Ե���ʽ���ݡ�֧������ѡ�
    %       'PenaltyFactor' - �ͷ�ϵ�� alpha��Ĭ��Ϊ2000����С��Լ��IMF�Ĵ���
    %       'LMUpdateRate' - ������ϵ�� tau��Ĭ��Ϊ0.01��
    %       'ExpectedIMFd' - Ԥ�ڵ�IMFģʽ��Ĭ��Ϊ�ա�����ʵ��
    %       'MaxIterations' - ������������Ĭ��Ϊ500��
    %       'AbsoluteTolerance' - �����ݲĬ��Ϊ1e-3��
    %       'CentralFrequencies' - Ԥ������ֽ��IMF������Ƶ�ʡ�
    %
    % �������:
    %   IMFd - �ֽ�õ�������Ƶ��ΪIMF
    %   centralFreqs - ÿ��IMF��Ӧ������Ƶ�ʡ�
    %   residuals - �в��ʾԭʼ�ź�������IMF֮��֮��Ĳ��졣
    %
    % ʾ������:
    %   [imf, freqs, res] = vme(signal, fs, 'NumIMFd', 4, 'PenaltyFactor', 1500);

%% ��ʼ��
[sgn,info] = setInfo(rawSignal,sampleRate,varargin);

%����ԭ�ź�Ƶ�� �Ⱦ��񲹳��˵�ЧӦ ��fft��ȡ���Ƶ��(ʵ�ź�Ƶ�׶Գƹ�ȡ���)
sgnFDfull = mirrorSignal(sgn,info,'add');%Frequency Domian
sgnFD = sgnFDfull(1:info.halfFreLen);

%��ʼ��ģ̬ ��ʼ��Ϊ0 ������Ƶ��(���)
initIMFd = zeros(info.rawSignLen,1);
initIMFdFDfull = fft(initIMFd,info.fftLen);
initIMFdFD = initIMFdFDfull(1:info.halfFreLen,:);
initIMFdFDNorm = abs(initIMFdFD).^2;

%��ʼ������
lambda =  zeros(size(sgnFD));
norFre = ((0:(info.fftLen/2))/info.fftLen).';%��һ��Ƶ����[0,0.5](�ź����Ƶ��Ϊ����Ƶ��һ��),��ʾƵ��
omega = info.centralFres;%��ʼ��������ȡģ̬����Ƶ��
iter = 0;
tolerance = inf;

%��������ڱ�����ֵ ��¼��ǰ���ϴε������
IMFdFD = initIMFdFD;
IMFdFDNorm = zeros(size(initIMFdFDNorm));%���ÿ��ģ̬��ƽ��
lastIMFdFD = IMFdFD;
%% ��ѭ��
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
%% ������
if (tolerance > info.diff)
    warning('VMD failed to converge!');%�ֽ�ʧ��
end
IMFdFDfull = complex(zeros(info.fftLen,1));%�Էֽ����ģ̬������չ �����ת��ʱ��
IMFdFDfull(1:info.halfFreLen,:) = IMFdFD;
IMFdFDfull(info.halfFreLen+1:end,:) = conj(IMFdFD(end-1:-1:2,:));%ʵ�ź���fft����Գ�

IMFd = mirrorSignal(IMFdFDfull,info,'remove');
centralFreqs = omega*sampleRate;
residuals = rawSignal - IMFd;
residuals = tolerance;
end

%% �����źź��� ���ٶ˵�ЧӦ
%  ��ԭ�źŲ������ ǰһ�뷭ת������ԭ�ź�ǰ�� ��һ�뷭ת������ԭ�źź���
function y = mirrorSignal(x,info,operate)
    if strcmp(operate, 'remove')
        xr = real(ifft(x));
        y = xr(info.halfrawSignLen+1:info.mirSignLen-info.halfrawSignLen,:);
    elseif strcmp(operate, 'add')
        xm = [x(info.halfrawSignLen:-1:1); x; x(info.rawSignLen:-1:info.halfrawSignLen+1)];
        y = fft(xm);
    end
end

%% Ĭ�ϲ�������
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
    for i = 1:2:length(varargin{1})-1 %vsd�д����vararginΪ1*n��cell����varargin�ٴ���Ϊ��������Ϊ1*1��cell���������{1*n��cell}
        varKey = varargin{1}{i};
        varVal = varargin{1}{i+1};
        switch varKey
            case 'PenaltyFactor'
                info.alpha = varVal;
            case 'LMUpdateRate'
                info.tau = varVal;
            case 'ExpectedIMFd' %Ԥ��ģ̬
                info.expectedIMFd = varVal;%������
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
    info.halfrawSignLen = length(sgn)/2;%����ָ��źų���
    info.mirSignLen = info.rawSignLen + info.halfrawSignLen*2;%�����ź��ܳ���
    info.fftLen = info.mirSignLen;
    info.halfFreLen = info.fftLen/2 + 1;
end
    