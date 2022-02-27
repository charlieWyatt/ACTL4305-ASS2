library(readr) 
library(dplyr)
library(randomForest)
library(caret) # Good for using CV as well as trees + RF + bagging
library(pROC)
library(ROCR)
library(tidyverse)
library(rpart.plot)
library(vip)
library(pdp)
library(EnvStats)
library(corrplot)
library(ROSE)

# https://topepo.github.io/caret/model-training-and-tuning.html
# https://bradleyboehmke.github.io/HOML/bagging.html

setwd("~/University/year3/T3/ACTL4305/Assignment/Assignment 2")

data <- read_csv("A2-data.csv", na = "empty")[,-1] 

data=data%>%mutate(pure.prem=claim.incurred/exposure) 

# Make some variables factors
factor_cols <- c("business.type", "driver.gender", "marital.status", "ncd.level", "region", "body.code", "fuel.type")
data <- data %>% 
  mutate_at(vars(factor_cols), funs(factor)) # makes all these factor variables

set.seed(654321) 

train.index=createDataPartition(data$claim.incurred, p = 0.7, list = FALSE)

train=data[train.index,] 

test=data[-train.index,] 

train <- train %>% select(-c(pure.prem, claim.count, region)) # removing claim.count since we dont have this at the start of reporting period
test <- test %>% select(-c(pure.prem, claim.count, region))

numeric_cols <- unlist(lapply(train, is.numeric))
numeric_data <- na.omit(train[,numeric_cols])
cor_data <- cor(numeric_data)
corrplot(cor_data, method = "circle")

# most correlated
cor_data[which(colnames(cor_data) == "weight"), which(colnames(cor_data) == "length")]
# weight is removed
cor_data <- cor_data[-which(colnames(cor_data) == "weight"), -which(colnames(cor_data) == "weight")]
corrplot(cor_data, method = "circle")

# percentage of 0 claim entries
sum(train$claim.incurred == 0)/length(train$claim.incurred)
# NEED TO FIX FOR UNBALANCED DATA -> MOST WILL BE 0


# attempt without caret and balanced
xtrain <- train %>% mutate("claimed" = as.factor(as.integer((claim.incurred != 0))))

xtrain <- ROSE(claimed ~ ., data = xtrain, seed = 1)$data # use ROSE to balance the data (default needs to be a factor for this)
xtrainClass <- xtrain %>% select(-claim.incurred)

rpartRawClass <- rpart(claimed ~ ., data = xtrainClass)
rpartRawReg <- rpart(claim.incurred~., data = filter(train, claim.incurred > 0), 
                     control = rpart.control(xval = 10, minbucket = 2, cp = 0))

printcp(rpartRawReg)
fit <- prune(rpartRawReg, cp = 0.02)
par(mar = rep(0.1, 4))
plot(fit, branch = 0.3, compress = TRUE)
text(fit)

rpart.plot(rpartRawReg) # requires rpart.plot package
title("Rpart Decision Tree plot")


































# Caret (unbalanced)

fitControl <- trainControl(
    method = "cv", 
    number = 5, 
    allowParallel = TRUE) # parrallel computing saves run time

rpart.grid <- expand.grid(cp=seq(0,0.01,0.001))
# Should test a number of cp values 
rpart <- train(claim.incurred ~., 
               data = train, 
               method = "rpart",
               trControl = fitControl,
               tuneGrid=rpart.grid)
# Standard r trees
  # plot(model$finalModel)
  # text(model$finalModel


rpart.plot(rpart$finalModel) # requires rpart.plot package
title("Rpart Decision Tree plot")

plot(rpart$results$cp, rpart$results$RMSE, type = "l", xlab = "cp", ylab = "RMSE", main = "Rpart RMSE changes for cp")
points(rpart$results$cp[which.min(rpart$results$RMSE)], min(rpart$results$RMSE), col = "red", pch = 16)

bestcp <- rpart$results$cp[which.min(rpart$results$RMSE)]

# Random Forest (do hyperparamter choosing here where m and number of nodes are tuned )
#hyper parameters
# number of trees
# deep of trees
# number of paramters (m)
# Use OOB error for these

{
  rf <- randomForest(formula = claim.incurred ~., data = train[1:100,],  importance = TRUE)
  
  varImpPlot(rf, main = "Feature Importance")
  
  
  # OOB vs test error
  index <- createDataPartition(train$claim.incurred, p = 0.75, list = FALSE)
  
  trainv <-train[index, ]; testv <- train[-index, ]
  
  x_test <- testv[,-which(colnames(testv) == "claim.incurred")]
  y_test<- testv$claim.incurred
  # random forest
  random_oob <- randomForest(claim.incurred ~., data = trainv, xtest = x_test, ytest = y_test)
  
  # extract OOB & validation errors
  oob <- random_oob$mse
  validation <-random_oob$test$mse
  # compare error rates
  tibble::tibble(`Out of Bag Error`= oob,`Test error`= validation,ntrees = 1:random_oob$ntree)%>%
    gather(Metric, Error, -ntrees) %>%
    ggplot(aes(ntrees, Error, color = Metric)) +  
    geom_line()+
    xlab("Number of trees")
}


mtry <- sqrt(ncol(train))
# finds the best m for RF by taking a range around the sqrt of number of predictors (usually considered a good approx)
rfGrid <-  expand.grid(.mtry=c(floor(mtry-2):ceiling(mtry+13)))


# HAVENT CHOSEN THE BEST NUMBER OF TREES! DO THIS FIRST!!!

# remove factors with too many levels
# dont use formula


# use ntree = c(200,500,1000) for random forest but this will be slower
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)


# rf <- train(claim.incurred ~ ., data = train, 
#              method = "rf", 
#              trControl = fitControl, 
#              verbose = FALSE,
#              tuneGrid = rfGrid)

rfTest <- train(claim.incurred ~ ., data = train, 
            method = "rf", 
            trControl = fitControl, 
            verbose = FALSE,
            tuneGrid = rfGrid,
            ntree = 25)

random_oob <- randomForest(formula = claim.incurred ~., 
                           data = train, 
                           xtest = select(test, -claim.incurred),
                           ytest = test$claim.incurred,
                           importance = TRUE
                           )

oob <- random_oob$err.rate[,1]
validation <-random_oob$test$err.rate[,1]

tibble::tibble(`Out of Bag Error`= oob,`Test error`= validation,ntrees = 1:random_oob$ntree)%>%
  gather(Metric, Error, -ntrees) %>%
  ggplot(aes(ntrees, Error, color = Metric)) +  
  geom_line()+
  xlab("Number of trees")


stopCluster(cluster)
registerDoSEQ()

plot(rf$results$mtry, rf$results$RMSE, type = "l", xlab = "mtry", ylab = "RMSE", main = "Random Forest RMSE changes for m")
points(rf$results$mtry[which.min(rf$results$RMSE)], min(rf$results$RMSE), col = "red", pch = 16)

vip(rf$finalModel)

# BAGGING

# No tuning paramters for this
# bagGrid <- expand.grid(.nbagg=(1:30)*50)

# NEED TO TUNE BEST NUMBER OF TREES HERE

treebag <- train(claim.incurred ~ ., data = train, 
                 method = "treebag", 
                 trControl = fitControl,
                 ntree = 25, 
                 cp = bestcp)


vip(treebag) # requires vip package


partial(treebag, 
        pred.var = "claim.count", 
        grid.resolution = 20) %>% 
  autoplot()



results <- test %>% select(claim.incurred)
results$predRpart <- predict(rpart, newdata = test)
results$predRf <- predict(rfTest, newdata = test)
results$predtreebag <- predict(treebag, newdata = test)






##### Balaced data

# Since ROSE doesnt work on regression problems 
# - a potential option is to make two classes of claim vs no claim then
# train a random forest on that classification problem with ROSE
# Then train a regression random forest tree on that.

xtrain <- train %>% mutate("claimed" = as.factor(as.integer((claim.incurred != 0))))

xtrain <- ROSE(claimed ~ ., data = xtrain, seed = 1)$data # use ROSE to balance the data (default needs to be a factor for this)

# Train wether it can classify claim occurences first
xClaimedTrain <- xtrain %>% select(-claim.incurred)

library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

xClaimRf <- train(claimed ~ ., data = xClaimedTrain, 
                method = "rf", 
                trControl = fitControl, 
                verbose = FALSE,
                tuneGrid = rfGrid,
                ntree = 25)

stopCluster(cluster)
registerDoSEQ()

plot(xClaimRf$results$mtry, xClaimRf$results$Accuracy, type = "l", xlab = "mtry", ylab = "Accuracy", main = "Random Forest RMSE changes for m")
points(xClaimRf$results$mtry[which.min(xClaimRf$results$Accuracy)], min(xClaimRf$results$Accuracy), col = "red", pch = 16)

vip(xClaimRf$finalModel)


# Now we have random forest model on claim or not
# now do random forest on amount incurred

IncTrain <- train %>% filter(claim.incurred > 0)

cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

IncRf <- train(claim.incurred ~ ., data = IncTrain, 
                method = "rf", 
                trControl = fitControl, 
                verbose = FALSE,
                tuneGrid = rfGrid,
                ntree = 25)

stopCluster(cluster)
registerDoSEQ()

plot(IncRf$results$mtry, IncRf$results$RMSE, type = "l", xlab = "mtry", ylab = "RMSE", main = "Random Forest RMSE changes for m")
points(IncRf$results$mtry[which.min(IncRf$results$RMSE)], min(IncRf$results$RMSE), col = "red", pch = 16)

vip(IncRf$finalModel)


# Then combine both xClaimRf and IncRf on test data to see how it goes
xtest <- test
xtest$claimed <- predict(xClaimRf, newdata = xtest)

predict(IncRf, filter(xtest, claimed == 1))








xIncurredTrain <- xtrain %>% filter(claimed == 1) %>% select(-claimed)

cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

xIncRf <- train(claim.incurred ~ ., data = xIncurredTrain, 
             method = "rf", 
             trControl = fitControl, 
             verbose = FALSE,
             tuneGrid = rfGrid,
             ntree = 25)

stopCluster(cluster)
registerDoSEQ()

plot(xIncRf$results$mtry, xIncRf$results$RMSE, type = "l", xlab = "mtry", ylab = "RMSE", main = "Random Forest RMSE changes for m")
points(xIncRf$results$mtry[which.min(xIncRf$results$RMSE)], min(xIncRf$results$RMSE), col = "red", pch = 16)

vip(xIncRf$finalModel)







### I SHOULD CHECK WETHER OR NOT I NEED THE ABOVE... OR IF ONE I USE ROSE TO SYNTHETICALLY OVERSAMPLE FROM CLASSES
### I CAN JUST USE THAT STRAIGHT AWAY. Or if that will introduce too much bias if the estimates are too unstable

# ISSUE! ROSE creates negative values for claim.incurred
# maybe could do a ROSE rf for classification, then a classification tree on the data I have for the regression

# https://journals.sagepub.com/doi/full/10.1177/0962280219888741 <- use this!!!

# straight away option is here-

xtrain <- xtrain %>% select(-claimed) # remove claimed now that synthetic samples are produced

cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

xrf <- train(claim.incurred ~ ., data = xtrain, 
             method = "rf", 
             trControl = fitControl, 
             verbose = FALSE,
             tuneGrid = rfGrid,
             ntree = 25)

stopCluster(cluster)
registerDoSEQ()

plot(xrf$results$mtry, xrf$results$RMSE, type = "l", xlab = "mtry", ylab = "RMSE", main = "Random Forest RMSE changes for m")
points(xrf$results$mtry[which.min(xrf$results$RMSE)], min(xrf$results$RMSE), col = "red", pch = 16)

vip(xrf$finalModel)


ROSEpred <- predict(xrf, newdata = test[,-c(22,23, 24)], type = "raw")



# GBM model

gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:30)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

gbm <- train(claim.incurred ~ ., data = train, 
             method = "gbm", 
             trControl = fitControl, 
             verbose = FALSE,
             tuneGrid = gbmGrid)

ggplot(gbm) +
  ggtitle("GBM Plot")

trellis.par.set(caretTheme())
plot(gbm)  

trellis.par.set(caretTheme())
plot(gbm, metric = "Rsquared")

trellis.par.set(caretTheme())
plot(gbm, plotType = "level",
     scales = list(x = list(rot = 90)))


# Output table of best models from RF, Bagged, tree and GBM




# Eh I dont like these that much
predTable <- treebag$pred %>% arrange(obs)
# not a particularly helpful summary of the error
  # SHOULD DO A COL GRAPH HERE OF THE COUNTS
tibble::tibble(`Prediction`= predTable$pred,`Observation`= predTable$obs,Entry = 1:length(predTable$pred))%>%
  gather(Class, `Pure Premium`, -Entry) %>%
  ggplot(aes(Entry, `Pure Premium`, color = Class)) +  
  geom_line()+
  xlab("Policy Holder")

# probably not helpful
# A comparison of the pdf plots of the observed vs expected with all 0 values removed
z <- predTable %>% filter(obs != 0)
epdfPlot(z$pred)
epdfPlot(z$obs)
