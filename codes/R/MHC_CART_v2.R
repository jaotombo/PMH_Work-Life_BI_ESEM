
setwd("F:/Dropbox/Projets/Work-Life Flourishing/Bi ESEM/R")

df = read.csv("WorkLife_Fixed_4cl_CART.csv", encoding="UTF-8",na.strings="*")

# Using CART and Machine Learning for predicting class assignment

MHC_data = df[,c(2,27:34)]
summary(MHC_data)

#======================================
# Definition of outcome and predictors
#======================================

outcomeName = "CL_MHC"
predNames = names(MHC_data)[!names(MHC_data) %in% outcomeName]

#===============================================================
#====   Pre processing
#===============================================================

# Loading caret package
library("caret")

# Check structure of data
str(MHC_data,list.len = 1000)
summary(MHC_data)
sum(is.na(MHC_data))

# set up training and testing data stratified on the outcome
table(MHC_data[,outcomeName])
prop.table(table(MHC_data[,outcomeName]))

# imput missing data
preProcValues = preProcess(MHC_data[,-1], method = "bagImpute")

library('RANN')
train = predict(preProcValues, MHC_data[,-1])
train[,outcomeName] = MHC_data[,outcomeName]
sum(is.na(train))

#============================
# === START RUNNING HERE ===
#============================

#Looking at the structure of the train data.
str(train)
summary(train)
table(train[,outcomeName])
prop.table(table(train[,outcomeName]))

# converting the outcome to a factor
train[,outcomeName] = as.factor(train[,outcomeName])

#===============================================================
# No Model Selection Analysis
#===============================================================

predictors = names(train)[!names(train) %in% outcomeName]


# Tuning parameters
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5 ,
  repeats = 5 ,
  classProbs = TRUE)

levels(train[,outcomeName]) = c("FL","MMH","FF","PL")

# FL = Fully Languishing
# MMH = Moderately Mentally Healthy
# Fully Flourishing
# Professionally Languishing

#========================================
# = Loop for averaging over the indices =
#========================================


library("caret")
library("doSNOW")
cl <- makeCluster(14, type = "SOCK")
registerDoSNOW(cl)

library(hmeasure)

#==============================================================
# CARET machine learning using CART 
#==============================================================

#-----------------------
# Predictions loop
#-----------------------
confusion_mat = c()
cart_list = list()

for (i in 1:100) {

  index <- createDataPartition(train[,outcomeName], p=0.7, list=FALSE)
  trainSet = train[index,]
  testSet = train[-index,]
  
  
  # CART define model
  model_cart1 = train(trainSet[,predictors],trainSet[,outcomeName],method='rpart',
                      trControl=fitControl, tuneLength=10)
  # CART predic model
  cart_pred = predict(model_cart1, testSet, type="raw")
  confusion = confusionMatrix(data = cart_pred, reference = testSet[,outcomeName])
  confusion_mat = rbind(confusion_mat, confusion$overall)
  cart_list[[i]] = model_cart1

}
#-------------------------------------------------------------------------------

confusion_perf = colMeans(confusion_mat)
print(confusion_perf[1:2])


#---------------------------------------------------------------
# End of parallel processing
stopCluster(cl)

# Tree for the optimal model :
par(mar=c(1,0,1,0))
best_cart = cart_list[[which.max(confusion_mat[,1])]]
plot(best_cart$finalModel, branch=1, uniform=TRUE, compress=TRUE, margin = 0)
text(best_cart$finalModel)


