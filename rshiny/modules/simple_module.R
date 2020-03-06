library("shiny")

counterButton <- function(id, label = "Conuter") {
  ns <- NS(id) # encapsulate module UI
  tagList(
    actionButton(ns("button"), label = label),
    verbatimTextOutput(ns("out"))
  )
}

counter <- function(input, output, session) {
  # Module server side logic, returns reactive value
  count <- reactiveVal(0)
  observeEvent(input$button, {
    count(count() + 1)
  })
  output$out <- renderText({
    count()
  })
  count
}

ui <- fluidPage(
  counterButton("counter1", "Counter #1")
)

server <- function(input, output, session){
  # use callModule to initialize
  callModule(counter, "counter1")
}

shinyApp(ui, server)

