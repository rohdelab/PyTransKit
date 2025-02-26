%% 1. translate template to center of image
clear all
filename = ''; %load filename 
addpath('View images/')
matfolder = (strcat(filename,'')) %folder with the images saved as matfiles
cd(filename);
b = dir(fullfile(matfolder,'*.mat'));
n = numel(b);
numfiles1 = length(b);
template = load(strcat(filename,'template.mat')) %load the template image
template = template.image;
template = thresh_template(template); %threshold template to create sharp edges
templatebin = imbinarize(templatethresh); %binarize template 
stats1 = regionprops3(templatebin, 'Centroid', 'Volume'); %find center of mass and volume of template
centroidt = stats1.Centroid;
volumet = stats1.Volume;
xdim = size(template, 1);
ydim = size(template, 2);
xdim = size(template, 3);
translationt = [centroidt(2)-xdim/2, centroidt(1)-ydim/2, centroidt(3)-zdim/2]; %generate translation vector to center of image, with reference to the image dimensions
translationt = translationt*-1 %invert translation vector
translationt = [translationt(2), translationt(1), translationt(3)]
translatedtemplate = imtranslate(template, translationt); %translate template
%imshow3D(translatedtemplate); %visualize the translated template 
translatedtemplatebin = imtranslate(templatebin, translationt);
newcentroid = regionprops3(translatedtemplatebin, 'Centroid'); %find center of mass of template
newcentroid = newcentroid.Centroid; %sanity check
%% 2. translate images to center of template and scale
scaledtfolder = (strcat(filename, '')) %if you wish to save scaled and translated images
scaledfolder = (strcat(filename,'Scaledonly/')) %if you wish to save scaled only images 
for i=1:numfiles1
    file1 = (strcat(matfolder, b(i).name));
    [filepath,name,ext] = fileparts(file1)
    newname = strsplit(name, '.')
    originalimage = load(file1); 
    originalimage = originalimage.image;
    originalimagebin = imbinarize(originalimage); %binarize image 
    stats2 = regionprops3(originalimagebin, 'Centroid', 'Volume'); %find center of mass and volume of image
    centroid = stats2.Centroid;
    centroidi = centroid(1,:)
    volume = stats2.Volume;
    volumei = volume(1,1)
    translationb = [centroidi(1)-newcentroid(1), centroidi(2)-newcentroid(2), centroidi(3)-newcentroid(3)];%generate translation vector to center of template, with reference to the template
    translationi = translationb*-1
    translatedimage = imtranslate(originalimage, translationi); %translate image
    %figure;imshow3D(translatedimage); %visualize the translated image
    sf = (volumet/volumei); %define scale factor
    sf = nthroot(sf, 3);%cube root because we are dealing with volume
    resizedimage = imresize3(translatedimage, sf, 'linear'); %resize image
    %figure;imshow3D(resizedimage);%show resized image
    targetsize = size(translatedimage);%crop or pad the resized image to correct dimensions
    resized = size(resizedimage)
    if sf>=1
        win = centerCropWindow3d(size(resizedimage), targetsize);
        scaled = imcrop3(resizedimage, win);
    else
        Xpadpr = floor((targetsize(1)-resized(1))/2);
        Ypadpr = floor((targetsize(2)-resized(2))/2);
        Zpadpr = floor((targetsize(3)-resized(3))/2);
        Xpadpo = round((targetsize(1)-resized(1))/2);
        Ypadpo = round((targetsize(2)-resized(2))/2);
        Zpadpo = round((targetsize(3)-resized(3))/2);
        scaled = padarray(resizedimage, [Xpadpr, Ypadpr, Zpadpr], 0, 'pre');
        scaled = padarray(scaled, [Xpadpo, Ypadpo, Zpadpo], 0, 'post');
    end
    translatedback = imtranslate(scaled, translationb); %translate the image back to original coordinates
    %figure;imshow3D(scaledcrop)
    save(strcat(scaledtfolder, newname{1}, '.mat'), 'scaled', 'translationi', 'sf') %savescaledandtranslated
    save(strcat(scaledfolder, newname{1}, '.mat'), 'translatedback', 'sf') %savescaledonly
    clear 'scaled' 'translatedback' 'translatedimage' 'originalimage'
end
%% 3. translate images to center of template without scaling
translatedfolder = (strcat(filename,'')) %if you wish to save translated images only
matfolder = (strcat(filename,'')) %folder with the images saved as matfiles
template = load(strcat(filename,'template.mat')) %load the template image
template = template.image;
template = thresh_template(template); %threshold template to create sharp edges
templatebin = imbinarize(templatethresh); %binarize template 
stats1 = regionprops3(templatebin, 'Centroid', 'Volume'); %find center of mass and volume of template
centroidt = stats1.Centroid;
for i=1:numfiles1
    file1 = (strcat(matfolder, b(i).name));
    [filepath,name,ext] = fileparts(file1)
    newname = strsplit(name, '.')
    originalimage = load(file1); 
    originalimage = originalimage.image;
    originalimagebin = imbinarize(originalimage); %binarize image 
    stats2 = regionprops3(originalimagebin, 'Centroid', 'Volume'); %find center of mass and volume of image
    centroid = stats2.Centroid;
    centroidi = centroid(1,:)
    translationb = [centroidi(1)-centroidt(1), centroidi(2)-centroidt(2), centroidi(3)-centroidt(3)];%generate translation vector to center of template, with reference to the template
    translationi = translationb*-1 %invert translation vector
    translatedimage = imtranslate(originalimage, translationi);
    save(strcat(translatedfolder, newname{1}, '.mat'), 'translatedimage', 'translationi')%save translation
    clear 'translatedimage' 'translationi'
end