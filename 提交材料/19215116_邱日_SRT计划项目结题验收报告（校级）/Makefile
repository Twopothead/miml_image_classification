my_thesis  := my
name       := 19215116_邱日_SRT报告
.PHONY	: clean run pdf

run: 
	xelatex -shell-escape -8bit $(my_thesis).tex -o $(my_thesis).pdf　
	rm  $(my_thesis).pdf
	xelatex -shell-escape -8bit $(my_thesis).tex -o $(my_thesis).pdf 
	@##要两次编译，才能得到正确的目录，由于\usepackage{minted} % 使用代码高亮，必须-shell-escape(要放在文件名前面)
	@##有8bit选项是为了代码中tab键不打印出奇怪字符
	rm *.aux
	mv $(my_thesis).pdf $(name).pdf

clean:
	if [ -f $(my_thesis).pdf ];then rm $(my_thesis).pdf;fi
	if [ -f $(my_thesis).toc ];then rm $(my_thesis).toc;fi
	if [ -f $(my_thesis).log ];then rm $(my_thesis).log;fi
	if [ -f $(my_thesis).aux ];then rm $(my_thesis).aux;fi
	if [ -f $(my_thesis).bib ];then rm $(my_thesis).bib;fi
	if [ -f $(my_thesis).out ];then rm $(my_thesis).out;fi
	if [ -f $(my_thesis).pyg ];then rm $(my_thesis).pyg;fi
	if [ -f $(my_thesis).log ];then rm $(my_thesis).log;fi
	
pdf:
	make run	
