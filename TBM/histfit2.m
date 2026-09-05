function h = histfit(varargin)
%HISTFIT Histogram with superimposed fitted normal density.
%   HISTFIT(DATA,NBINS) plots a histogram of the values in the vector DATA,
%   along with a normal density function with parameters estimated from the
%   data.  NBINS is the number of bars in the histogram. With one input
%   argument, NBINS is set to the square root of the number of elements in
%   DATA. 
%
%   HISTFIT(AX,...) plots into AX instead of GCA.
% 
%   HISTFIT(DATA,NBINS,DIST) plots a histogram with a density from the DIST
%   distribution.  DIST can take the following values:
%
%         'beta'                             Beta
%         'birnbaumsaunders'                 Birnbaum-Saunders
%         'exponential'                      Exponential
%         'extreme value' or 'ev'            Extreme value
%         'gamma'                            Gamma
%         'generalized extreme value' 'gev'  Generalized extreme value
%         'generalized pareto' or 'gp'       Generalized Pareto (threshold 0)
%         'inverse gaussian'                 Inverse Gaussian
%         'logistic'                         Logistic
%         'loglogistic'                      Log logistic
%         'lognormal'                        Lognormal
%         'negative binomial' or 'nbin'      Negative binomial
%         'nakagami'                         Nakagami
%         'normal'                           Normal
%         'poisson'                          Poisson
%         'rayleigh'                         Rayleigh
%         'rician'                           Rician
%         'tlocationscale'                   t location-scale
%         'weibull' or 'wbl'                 Weibull
%
%   H = HISTFIT(...) returns a vector of handles to the plotted lines.
%   H(1) is a handle to the histogram, H(2) is a handle to the density curve.

%   Copyright 1993-2020 The MathWorks, Inc.

% Allow uipanel/figure input as the first argument (parent)
nin = nargin;
if isempty(varargin)
    error(message('stats:histfit:VectorRequired'));
elseif isa(varargin{1},'matlab.graphics.axis.Axes')
    ax = varargin{1};
    varargin = varargin(2:end);
    nin = nin - 1;
else
    ax = newplot;
end

if nin > 2 % nin == 3, all params case
    [~,nbins,dist] = varargin{:};
    dist = convertStringsToChars(dist);
elseif nin == 2 
    dist=[];nbins = varargin{2};
else
    nbins=[];dist=[];
end

if nin <= 0
   error(message('stats:histfit:VectorRequired'));
end

data = varargin{1};
if ~isvector(data)
   error(message('stats:histfit:VectorRequired'));
end

data = data(:);
data(isnan(data)) = [];
n = numel(data);

if nin<2 || isempty(nbins) % nin == 1, give nbins value
    nbins = ceil(sqrt(n));
elseif ~isscalar(nbins) || ~isnumeric(nbins) || ~isfinite(nbins) ...
                        || nbins~=round(nbins) || nbins<=0
    error(message('stats:histfit:BadNumBins'))
end

% Fit distribution to data
if nin<3 || isempty(dist) % nin == 1 and nin == 2, give dist value
    dist = 'normal';
end
try
    pd = fitdist(data,dist);
catch myException
    if isequal(myException.identifier,'stats:ProbDistUnivParam:fit:NRequired') || ...
       isequal(myException.identifier,'stats:binofit:InvalidN')
        % Binomial is not allowed because we have no N parameter
        error(message('stats:histfit:BadDistribution'))
    else
        % Pass along another other errors
        throw(myException)
    end
end

% Find range for plotting
q = icdf(pd,[0.0013499 0.99865]); % three-sigma range for normal distribution
x = linspace(q(1),q(2));
if ~pd.Support.iscontinuous
    % For discrete distribution use only integers
    x = round(x);
    x(diff(x)==0) = [];
end

% Do histogram calculations
[bincounts,binedges] = histcounts(data,nbins);
bincenters = binedges(1:end-1)+diff(binedges)/2;

% Plot the histogram with no gap between bars.
%hh = bar(ax, bincenters,bincounts,1);

% Normalize the density to match the total area of the histogram
binwidth = binedges(2)-binedges(1); % Finds the width of each bin
area = n * binwidth;
y = area * pdf(pd,x);
y = y/sum(y(:))
% Overlay the density
np = get(ax,'NextPlot');
set(ax,'NextPlot','add');
hh1 = plot(ax, x,y,'r-','LineWidth',2);
c = [0.8500 0.3250 0.0980];
data1 = fill(x, y, c, 'FaceAlpha',0.8)
if nargout == 1
  h = [hh1];
end

set(ax,'NextPlot',np);