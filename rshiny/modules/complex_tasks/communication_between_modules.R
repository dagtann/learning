rm(list = ls())

library(shiny)
library(shinyWidgets)
library(ggplot2)

# Inner module setup ==========================================================
inner_module_ui <- function(id) {
  ns <- NS(id)
  
  tagList(
    plotOutput(ns("inner_plot")),
    materialSwitch(ns("export_switch"))
  )
}

inner_module <- function(input, output, session) {
  # preallocate object
  string_return <- character(1)
  
  plot_data <- data.frame(x = runif(10), y = runif(10))
  plot_object <- reactive({
    ggplot(data = plot_data, aes(x = x, y = y)) +
      geom_point() +
      labs(title = "I am inside the inner module")
  })
  output$inner_plot <- renderPlot({plot_object()})
  
  return_value <- eventReactive(input$export_switch, {
    input$export_switch
  })
  return(reactive(return_value()))
}

# Outer module setup ==========================================================
outer_module_ui <- function(id) {
  ns <- NS(id)
  
  fluidPage(
    inner_module_ui(ns("plot")),
    verbatimTextOutput(ns("text"))
  )
}

outer_module <- function(input, output, session) {
  inner_plot_return <- callModule(inner_module, "plot")

  output$text <- renderText({
    paste("I am reactive:", is.reactive(inner_plot_return()), "My value is:", inner_plot_return())
  })
}

# Shiny App functions =========================================================
ui <- fluidPage(
  outer_module_ui("outer")
)

server <- function(input, output, session) {
  callModule(outer_module, "outer")
}

shinyApp(ui, server)
