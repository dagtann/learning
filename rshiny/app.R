library("shiny")

ui <- fluidPage(
    tags$img(width = 100, height = 100, src = "my-avatar.png"),
    tags$h1("Watch Out!"),
    tags$em("This is a Shiny app"),
    tags$p("You would not have", tags$strong("noticed"))
)
server <- function(input, output) {

}
shinyApp(ui = ui, server = server)

# ui <- fluidPage(
#     tabsetPanel(
#         tabPanel("tab 1", "contents"),
#         tabPanel("tab 2", "contents"),
#         tabPanel("tab 3", "contents")
#     )
# )
# shinyApp(ui = ui, server = server)