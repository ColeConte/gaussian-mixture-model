'''
Cole Conte CSE 512 Hw 3
'''

def gmmLearner(train,test,components):
	pass

parser = argparse.ArgumentParser()
parser.add_argument("--components")
parser.add_argument("--train")
parser.add_argument("--test")

train = pd.read_csv(args.train)
test = pd.read_csv(args.test)
gmmLearner(train,test,int(components))


