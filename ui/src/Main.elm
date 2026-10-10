module Main exposing (main)

import Browser
import Html exposing (Html, button, div, h1, h2, header, input, label, li, ol, option, p, section, select, text)
import Html.Attributes exposing (class, disabled, placeholder, selected, style, type_, value)
import Html.Events exposing (onClick, onInput)
import Http
import Json.Decode as Decode
import Json.Encode as Encode
import Time


main : Program () Model Msg
main =
    Browser.element
        { init = \_ -> ( initial, Cmd.none )
        , view = view
        , update = update
        , subscriptions = subscriptions
        }


type alias Model =
    { location : String
    , limit : String
    , metric : String
    , index : String
    , phase : Phase
    , stage : String
    , lines : List String
    , done : Int
    , total : Maybe Int
    , error : Maybe String
    , ready : Maybe Summary
    , query : String
    , k : String
    , hits : List Hit
    , searching : Bool
    }


type Phase
    = Idle
    | Loading
    | Ready
    | Failed


type alias Summary =
    { rows : Int
    , dim : Int
    , buildMs : Int
    , metric : String
    , index : String
    , preview : List Hit
    }


type alias Hit =
    { id : String
    , distance : Maybe Float
    , title : String
    , text : String
    }


type alias Status =
    { phase : String
    , stage : String
    , lines : List String
    , done : Int
    , total : Maybe Int
    , error : Maybe String
    , ready : Maybe Summary
    }


initial : Model
initial =
    { location = ""
    , limit = "10000"
    , metric = "cosine"
    , index = "hnsw"
    , phase = Idle
    , stage = "idle"
    , lines = []
    , done = 0
    , total = Nothing
    , error = Nothing
    , ready = Nothing
    , query = "0"
    , k = "5"
    , hits = []
    , searching = False
    }


type Msg
    = EditLocation String
    | EditLimit String
    | ChooseMetric String
    | ChooseIndex String
    | EditQuery String
    | EditK String
    | ClickLoad
    | ClickSample
    | ClickSearch
    | LoadPosted (Result String ())
    | Tick Time.Posix
    | StatusArrived (Result Http.Error Status)
    | SearchArrived (Result Http.Error (List Hit))


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        EditLocation value ->
            ( { model | location = value }, Cmd.none )

        EditLimit value ->
            ( { model | limit = value }, Cmd.none )

        ChooseMetric value ->
            ( { model | metric = value }, Cmd.none )

        ChooseIndex value ->
            ( { model | index = value }, Cmd.none )

        EditQuery value ->
            ( { model | query = value }, Cmd.none )

        EditK value ->
            ( { model | k = value }, Cmd.none )

        ClickSample ->
            startLoad "sample://clusters" { model | location = "sample://clusters" }

        ClickLoad ->
            startLoad (String.trim model.location) model

        LoadPosted (Ok ()) ->
            ( model, fetchStatus )

        LoadPosted (Err reason) ->
            ( { model | phase = Failed, error = Just reason }, Cmd.none )

        Tick _ ->
            ( model, fetchStatus )

        StatusArrived (Ok status) ->
            ( applyStatus model status, Cmd.none )

        StatusArrived (Err err) ->
            ( { model | error = Just (httpError err) }, Cmd.none )

        ClickSearch ->
            case String.toInt model.k of
                Just k ->
                    ( { model | searching = True, error = Nothing }, postSearch model.query k )

                Nothing ->
                    ( { model | error = Just "k must be a whole number" }, Cmd.none )

        SearchArrived (Ok hits) ->
            ( { model | hits = hits, searching = False }, Cmd.none )

        SearchArrived (Err err) ->
            ( { model | searching = False, error = Just (httpError err) }, Cmd.none )


startLoad : String -> Model -> ( Model, Cmd Msg )
startLoad location model =
    case String.toInt model.limit of
        Nothing ->
            ( { model | error = Just "limit must be a whole number" }, Cmd.none )

        Just limit ->
            ( { model
                | phase = Loading
                , stage = "resolve"
                , lines = [ "starting" ]
                , done = 0
                , total = Just limit
                , error = Nothing
                , ready = Nothing
                , hits = []
              }
            , postLoad location limit model.metric model.index
            )


applyStatus : Model -> Status -> Model
applyStatus model status =
    { model
        | phase = phaseFrom status.phase
        , stage = status.stage
        , lines = status.lines
        , done = status.done
        , total = status.total
        , error = status.error
        , ready = status.ready
    }


phaseFrom : String -> Phase
phaseFrom name =
    case name of
        "loading" ->
            Loading

        "ready" ->
            Ready

        "failed" ->
            Failed

        _ ->
            Idle


subscriptions : Model -> Sub Msg
subscriptions model =
    if model.phase == Loading then
        Time.every 400 Tick

    else
        Sub.none


view : Model -> Html Msg
view model =
    div [ class "page" ]
        [ header []
            [ h1 [] [ text "vectordb" ]
            , p []
                [ text "Point this at a local embedding file or folder, or at an https URL. The page shows the parse and the index build, then you can search by row id." ]
            ]
        , section []
            [ label [] [ text "Dataset location" ]
            , input
                [ value model.location
                , placeholder "/path/to/dataset  or  https://example.com/embeddings.parquet"
                , onInput EditLocation
                , onEnter ClickLoad
                ]
                []
            , p [ class "hint" ]
                [ text "Parquet, csv, tsv, or jsonl. A folder is scanned a few levels down. Kaggle sets are large: download them, then paste the folder. sample://clusters is built in." ]
            , div [ class "row" ]
                [ div []
                    [ label [] [ text "Row limit" ]
                    , input [ value model.limit, onInput EditLimit ] []
                    ]
                , div []
                    [ label [] [ text "Metric" ]
                    , select [ onInput ChooseMetric ]
                        [ choice model.metric "cosine" "cosine"
                        , choice model.metric "l2" "l2"
                        , choice model.metric "ip" "inner product"
                        ]
                    ]
                ]
            , label [] [ text "Index" ]
            , select [ onInput ChooseIndex ]
                [ choice model.index "hnsw" "hnsw"
                , choice model.index "flat" "flat (exact)"
                ]
            , div [ class "actions" ]
                [ button [ type_ "button", onClick ClickLoad, disabled (model.phase == Loading) ] [ text "Load" ]
                , button [ type_ "button", class "secondary", onClick ClickSample, disabled (model.phase == Loading) ] [ text "Load sample" ]
                ]
            ]
        , progressView model
        , maybeError model.error
        , readyView model
        ]


choice : String -> String -> String -> Html Msg
choice current item labelText =
    option [ value item, selected (current == item) ] [ text labelText ]


progressView : Model -> Html Msg
progressView model =
    if model.phase == Idle && List.isEmpty model.lines then
        text ""

    else
        section []
            [ p [ class "stage" ] [ text (stageText model) ]
            , div [ class "bar" ] [ div [ style "width" (barWidth model) ] [] ]
            , ol [ class "log" ] (List.map (\line -> li [] [ text line ]) model.lines)
            ]


stageText : Model -> String
stageText model =
    case model.total of
        Just total ->
            model.stage ++ "  " ++ String.fromInt model.done ++ " / " ++ String.fromInt total

        Nothing ->
            model.stage


barWidth : Model -> String
barWidth model =
    case model.total of
        Just total ->
            if total <= 0 then
                "0%"

            else
                String.fromFloat (toFloat model.done / toFloat total * 100) ++ "%"

        Nothing ->
            "0%"


readyView : Model -> Html Msg
readyView model =
    case model.ready of
        Nothing ->
            text ""

        Just ready ->
            section []
                [ p [ class "meta" ]
                    [ text
                        (String.fromInt ready.rows
                            ++ " vectors, dim "
                            ++ String.fromInt ready.dim
                            ++ ", "
                            ++ ready.metric
                            ++ " / "
                            ++ ready.index
                            ++ ", built in "
                            ++ String.fromInt ready.buildMs
                            ++ " ms"
                        )
                    ]
                , p [ class "hint" ]
                    [ text "Distance is smaller when a row is nearer. Cosine and inner product are reported as a negative dot product." ]
                , label [] [ text "Query" ]
                , input
                    [ value model.query
                    , placeholder "row id, or 1, 0, 0, 0"
                    , onInput EditQuery
                    , onEnter ClickSearch
                    ]
                    []
                , label [] [ text "k" ]
                , input [ value model.k, onInput EditK ] []
                , div [ class "actions" ]
                    [ button
                        [ type_ "button"
                        , onClick ClickSearch
                        , disabled model.searching
                        ]
                        [ text
                            (if model.searching then
                                "Searching"

                             else
                                "Search"
                            )
                        ]
                    ]
                , h2 [] [ text "First rows" ]
                , div [ class "cards" ] (List.map (card False) ready.preview)
                , if List.isEmpty model.hits then
                    text ""

                  else
                    section []
                        [ h2 [] [ text "Nearest" ]
                        , div [ class "cards" ] (List.map (card True) model.hits)
                        ]
                ]


card : Bool -> Hit -> Html Msg
card showDistance hit =
    Html.article []
        [ p [ class "distance" ]
            [ text
                ("id "
                    ++ hit.id
                    ++ (if showDistance then
                            case hit.distance of
                                Just distance ->
                                    "   distance " ++ String.fromFloat distance

                                Nothing ->
                                    ""

                        else
                            ""
                       )
                )
            ]
        , h2 [] [ text hit.title ]
        , if hit.text == "" then
            text ""

          else
            p [] [ text hit.text ]
        ]


maybeError : Maybe String -> Html Msg
maybeError problem =
    case problem of
        Nothing ->
            text ""

        Just reason ->
            p [ class "error" ] [ text reason ]


onEnter : Msg -> Html.Attribute Msg
onEnter msg =
    Html.Events.on "keydown"
        (Decode.field "key" Decode.string
            |> Decode.andThen
                (\key ->
                    if key == "Enter" then
                        Decode.succeed msg

                    else
                        Decode.fail "not enter"
                )
        )


postLoad : String -> Int -> String -> String -> Cmd Msg
postLoad location limit metric index =
    Http.post
        { url = "/api/load"
        , body =
            Http.jsonBody
                (Encode.object
                    [ ( "location", Encode.string location )
                    , ( "limit", Encode.int limit )
                    , ( "metric", Encode.string metric )
                    , ( "index", Encode.string index )
                    ]
                )
        , expect = expectLoad
        }


expectLoad : Http.Expect Msg
expectLoad =
    Http.expectStringResponse LoadPosted <|
        \response ->
            case response of
                Http.BadUrl_ url ->
                    Err ("bad url " ++ url)

                Http.Timeout_ ->
                    Err "the request timed out"

                Http.NetworkError_ ->
                    Err "network error"

                Http.BadStatus_ _ body ->
                    Err (errorFromBody body)

                Http.GoodStatus_ _ _ ->
                    Ok ()


fetchStatus : Cmd Msg
fetchStatus =
    Http.get
        { url = "/api/status"
        , expect = Http.expectJson StatusArrived statusDecoder
        }


postSearch : String -> Int -> Cmd Msg
postSearch query k =
    Http.post
        { url = "/api/search"
        , body =
            Http.jsonBody
                (Encode.object
                    [ ( "query", Encode.string query )
                    , ( "k", Encode.int k )
                    ]
                )
        , expect = Http.expectJson SearchArrived (Decode.field "hits" (Decode.list hitDecoder))
        }


statusDecoder : Decode.Decoder Status
statusDecoder =
    Decode.map7 Status
        (Decode.field "phase" Decode.string)
        (Decode.field "stage" Decode.string)
        (Decode.field "lines" (Decode.list Decode.string))
        (Decode.field "done" Decode.int)
        (Decode.field "total" (Decode.nullable Decode.int))
        (Decode.field "error" (Decode.nullable Decode.string))
        (Decode.field "ready" (Decode.nullable readyDecoder))


readyDecoder : Decode.Decoder Summary
readyDecoder =
    Decode.map6 Summary
        (Decode.field "rows" Decode.int)
        (Decode.field "dim" Decode.int)
        (Decode.field "build_ms" Decode.int)
        (Decode.field "metric" Decode.string)
        (Decode.field "index" Decode.string)
        (Decode.field "preview" (Decode.list hitDecoder))


hitDecoder : Decode.Decoder Hit
hitDecoder =
    Decode.map4 Hit
        (Decode.field "id" Decode.string)
        (Decode.field "distance" (Decode.nullable Decode.float))
        (Decode.field "title" Decode.string)
        (Decode.field "text" Decode.string)


errorFromBody : String -> String
errorFromBody body =
    case Decode.decodeString (Decode.field "error" Decode.string) body of
        Ok message ->
            message

        Err _ ->
            if String.isEmpty body then
                "request failed"

            else
                body


httpError : Http.Error -> String
httpError err =
    case err of
        Http.BadUrl url ->
            "bad url " ++ url

        Http.Timeout ->
            "the request timed out"

        Http.NetworkError ->
            "network error"

        Http.BadStatus code ->
            "request failed (" ++ String.fromInt code ++ ")"

        Http.BadBody body ->
            errorFromBody body
