function nor_label = Normalize(InImg)
  ymax=255;ymin=0;
  xmax = max(max(InImg));
  xmin = min(min(InImg));
  nor_label = round((ymax-ymin)*(InImg-xmin)/(xmax-xmin) + ymin);
end
