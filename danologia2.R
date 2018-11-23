set.seed(1)
install.packages("caret")
install.packages("mlr")
install.packages("FSelector")
install.packages("scidb")
install.packages("dplyr")
install.packages("knitr")
install.packages("ParamHelpers")
install.packages("ggplot2")
install.packages("kernlab")
install.packages("mlrMBO")
install.packages("DiceKriging")

library(caret)
library(mlr)
library(FSelector)
library(scidb)
library(dplyr)
library(knitr)
library(ParamHelpers)
library(ggplot2)
library(kernlab)
library(digest)
library(parallelMap)
trainCsv <- read.csv('lucene2.2train.csv',header=TRUE,sep=",")
testCsv <- read.csv('lucene2.4test.csv',header=TRUE,sep=",")

mlr::summarizeColumns(trainCsv)

par(mfrow=c(1,4))
for (i in 5:8) {
  boxplot(trainCsv[,i], main=names(trainCsv)[i])
}

#split input and output
iv <- trainCsv[,5:8] #miary produktu 
dv <-  trainCsv[,29]
caret::featurePlot(x=iv,y=as.factor(dv),plot="box",
                   scales=list(x=list(relation="free"),y=list(relation="free")),
                   auto.key=list(columns=2))


#split input and output
iv <- trainCsv[,23:26] #miary procesów
dv <-  trainCsv[,29]
caret::featurePlot(x=iv,y=as.factor(dv),plot="box",
                   scales=list(x=list(relation="free"),y=list(relation="free")),
                   auto.key=list(columns=2))

#cleaning data - preprocessing
train <- trainCsv %>% mutate(dataset = "train")
test <- testCsv %>% mutate(dataset = "test")
combined <- dplyr::bind_rows(train,test)
mlr::summarizeColumns(combined) %>% kable(digits = 2)

combined <- combined %>% select(-c(X,Project,Version,Class)) 

imp <- mlr::impute(combined,
                   classes = list(#named list containing imputation techniques for classes of data
                     factor = mlr::imputeMode(), integer=mlr::imputeMean(), numeric=mlr::imputeMean()))
combined <- imp$data
mlr::summarizeColumns(combined) %>% kable(digits = 2) #show column summary

#normalizacja
combined <- mlr::normalizeFeatures(combined,target="isBuggy")
mlr::summarizeColumns(combined) %>% kable(digits = 2) #show column summary

#finishing preprocessing - rozdzielenie zbioru treningowego i testowego
train <- combined %>% filter(dataset=="train") %>% select(-dataset)
test <- combined %>% filter(dataset=="test") %>% select(-dataset)
mlr::summarizeColumns(train) %>% kable(digits = 2) #show column summary

#modelowanie
#create tasks
trainTask <- mlr::makeClassifTask(data=train,target="isBuggy",positive="TRUE")
testTask <- mlr::makeClassifTask(data=test,target="isBuggy",positive="TRUE")
print(trainTask)

im_feat <- mlr::generateFilterValuesData(trainTask, method = c("information.gain","chi.squared"))
mlr::plotFilterValues(im_feat,n.show = 20)

getParamSet("classif.ksvm")
ksvm.learner <- makeLearner("classif.ksvm", predict.type = "response")
paramSetSVM = makeParamSet(
  makeNumericParam("C", -10, 10, trafo = function(x) 2^x),
  makeNumericParam("epsilon", -10, 10, trafo = function(x) 2^x),
  makeNumericParam("sigma", -10, 10, trafo = function(x) 2^x)
)
set_cv = makeResampleDesc("CV",iters = 3L)
controlSVM <- makeTuneControlGrid()
res <- tuneParams(ksvm.learner, task = trainTask, resampling = set_cv, par.set = paramSetSVM, control = controlSVM,measures = mcc)
print(res)
parameters.svm <- mlr::train(ksvm.learner, trainTask)
predict.svm <- predict(parameters.svm, testTask)
perfMeasure <- mlr::performance(predict.svm, measures = list(mcc,mmce,acc,f1,kappa))
print(perfMeasure)
