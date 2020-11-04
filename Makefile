todo:
		ack -R 'TODO' {bin,lib,notebooks,notes,reports,services} || :

clean_dist:
		cd lib && rm -rf dist
 
dist: clean_dist
		cd lib && python setup.py sdist
 
pypi: dist
		cd lib && twine upload --repository pypi dist/*
 
