SLIC���÷�ʽ��
	��һ����
		sl.SLIC(image,step=20)
		sl.main()
	��
		sl.rgb2Lab():RGBתLab�ռ�
    		sl.seeds_init()�����ӵ��ʼ���㣻
    		sl.seeds_adjust()�����ӵ��ݶȵ�����
		sl.Seeds_grow():�����طָ�
	�ڶ�����
		rg=Region_grow(sl.dec_mat,count=100):sl.dec_matΪ������ͼ��,countΪ��							С��������Ԫ����;
		rg.main()
	��
		rg.grow()������������ȷ������Ҫ��ĳ����ظ��壨����С��������Ԫ���ϣ�
        	rg.enhance():����Ϊ����Ҫ����Ԫ�������·�����Ԫ��ǩ���ͽ�ԭ��
	������������ѡ��
		rg.makeboundary(template_img):���Ƴ����ر߽纯����template_imgΪģ��						ͼ��;