clean:
	rm -rf *~
	find ./ -depth -name ".AppleDouble" -exec rm -Rf {} \;
	find ./ -depth -name ".DS_Store" -exec rm -Rf {} \;
	find ./ -depth -name "__pycache__" -exec rm -Rf {} \;
