rm(list = ls())
library(shiny)
setwd("C:/Users/32856/Documents/git/learning/rshiny/modules/input_example")

if(!file.exists("mtcars.csv")) {
  write.csv(mtcars, "./mtcars.csv", row.names = FALSE)
}

source("csvFile.R")

ui <- fluidPage(
  sidebarLayout(
    sidebarPanel(
      csvFileInput("datafile", "User data (.csv format)")
    ),
    mainPanel(
      dataTableOutput("table")
    )
  )
)

server <- function(input, output, session) {
  datafile <- callModule(csvFile, "datafile", stringsAsFactors = FALSE)
  output$table <- renderDataTable({datafile()})
}

shinyApp(ui, server)