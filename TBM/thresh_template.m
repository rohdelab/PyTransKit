function[templatethresh] = thresh_template(template)
n_slices = size(template, 3);
for a=1:n_slices %threshold the template image to create sharp edges
    t=template(:,:,a);
    in=find(t<5);
    t(in)=0;
    templatethresh(:,:,a)=t;
    templatethresh(templatethresh<5)=0;
end