function nor_label = Normalize(InImg)
ymax=255;ymin=0;
xmax = max(max(InImg)); %���InImg�е����ֵ
xmin = min(min(InImg)); %���InImg�е���Сֵ
nor_label = round((ymax-ymin)*(InImg-xmin)/(xmax-xmin) + ymin); %��һ����ȡ��
end
