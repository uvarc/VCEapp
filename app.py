from flask import Flask, send_from_directory
from flask_compress import Compress

import pandas as pd
import numpy as np
import os
from io import BytesIO
import base64
import cv2
import time
#import dash_auth

import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_table
from dash.dependencies import Input, Output, State

import plotly
from plotly.subplots import make_subplots
import plotly_express as px
import plotly.graph_objects as go


from io import BytesIO
import base64
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

static_image_route = '/static/'
filespath='/project/DSone/jaj4zcf/Videos/'

import dash_table.FormatTemplate as FormatTemplate
from dash_table.Format import Sign

from dash.exceptions import PreventUpdate

import config

if config.live==False:
    import dbsqlite as dbf
else:
    import dbmysql as dbf
#import appFunctions as apfn
import datetime
from dash_extensions import Keyboard

print('packages loaded')
COMPRESS_MIMETYPES = ['text/html', 'text/css', 'text/xml', 'application/json', 'application/javascript', 'image/png', 'image/jpg']


app = Flask(__name__)

Compress(app)

print('app compressed')
application = dash.Dash(__name__, server=app,url_base_pathname='/')





def buttonOptions(inputList):
    outList=[]
    for item in inputList:
        outList.append({'label':item, 'value':item})
    return(outList)


##Function to Return Images for Table Below
def getImage(vid, frame):    
    impath='/project/DSone/jaj4zcf/Videos/v'+str(vid)[-2:]+'/'+str(frame)+'.jpg'    ## may need to be _rowd for final!
    impathdirect='/vids/v'+str(vid)[-2:]+'/'+str(frame)+'.jpg'

    if config.live==True:
        imgHTTP=impathdirect
    else:
        encoded_image = base64.b64encode(open(impath, 'rb').read()).decode("ascii").replace("\n", "")
        imgHTTP='data:image/jpg;base64,{}'.format(encoded_image)
        
    return imgHTTP


def buildTopTableStatic(sectOptions,abnormalOptions,frames):
    
    multi_sel_width= 3
    max_image_width = str(round((99-multi_sel_width) * 1/int(config.n_images), 1))+'%'  
    multi_sel_width= str(multi_sel_width)+'%'
    
    sectOpts=buttonOptions(sectOptions)
    abnormalOpts=buttonOptions(abnormalOptions)
    abnormal_multi_select_options=[{'label':'', 'value':item['value']} for item in abnormalOpts]
    images=[]
    labels=[]
    sectOptionsButts=[]
    abnormalOptionsButts=[]
    inflammation=[]
    edemous_villi=[]
    diffuse_bleeding=[]
    bleed=[]
    notes = []
    
    ## Add multiple select/notes - 
    images.append(html.Div(html.Img(id='imgmulti', style={'width':'100%', 'border-width': '2px','border-style': 'solid', 'border-color': 'white'}), 
                               style={'overflow':'hidden', 'max-width':multi_sel_width, 'width':'250px', 'display': 'inline-block'
                                     }, className='zoomlow'))
    labels.append(html.Div(html.P(children='all', id='all', style={'color':'black','font-size':'14px','height':'30px', 'background-color':'white','border-width': '2px','border-style': 'solid', 'border-color': 'white'}), 
                               style={'max-height': '30px', 'overflow':'hidden', 'max-width':multi_sel_width, 'width':'250px', 'display': 'inline-block'}))
    sectOptionsButts.append(html.Div(html.P(children=html.Button('c',id='clone'), id='all2', style={'color':'black','font-size':'14px','height':'30px', 'background-color':'white','border-width': '2px','border-style': 'solid', 'border-color': 'white'}), 
                               style={'max-height': '30px', 'overflow':'hidden', 'max-width':multi_sel_width, 'width':'250px', 'display': 'inline-block'}))
    
    abnormalOptionsButts.append(html.Div(dcc.RadioItems(id='abButt_multi',
            options=abnormal_multi_select_options, labelStyle={'display': 'block'}, style={'width':'100%', 'background-color': 'MistyRose','border-width': '2px','border-style': 'solid', 'border-color': 'white'}
            ), style={'max-width':multi_sel_width,  'width':'250px', 'display': 'inline-block'}))   
    inflammation.append(html.Div(dcc.Checklist(id='inflammation_multi',
            options=[{'label': '', 'value': 1}], labelStyle={'display': 'block'}, style={'width':'100%', 'background-color': 'RosyBrown','border-width': '2px','border-style': 'solid', 'border-color': 'white'}
            ), style={'max-width':multi_sel_width,  'width':'250px', 'display': 'inline-block'}))   
    edemous_villi.append(html.Div(dcc.Checklist(id='edemous_villi_multi',
            options=[{'label': '', 'value': 1}], labelStyle={'display': 'block'}, style={'width':'100%', 'background-color': 'Tan','border-width': '2px','border-style': 'solid', 'border-color': 'white'}
            ), style={'max-width':multi_sel_width,  'width':'250px', 'display': 'inline-block'})) 
    diffuse_bleeding.append(html.Div(dcc.Checklist(id='diffuse_multi',
            options=[{'label': '', 'value': 1}],  labelStyle={'display': 'block'}, style={'width':'100%', 'background-color': 'DarkSalmon','border-width': '2px','border-style': 'solid', 'border-color': 'white'}
            ), style={'max-width':multi_sel_width,  'width':'250px', 'display': 'inline-block'})) 
    bleed.append(html.Div(dcc.Checklist(id='bleed_multi',
            options=[{'label': '', 'value': 1}],  labelStyle={'display': 'block'}, style={'width':'100%', 'background-color': 'LightCoral','border-width': '2px','border-style': 'solid', 'border-color': 'white'}
            ), style={'max-width':multi_sel_width,  'width':'250px', 'display': 'inline-block'})) 
    notes.append(html.Div(dcc.Textarea(id='notes_multi',
            style={'width':'100%', 'background-color': '#fae5d3','border-width': '2px','border-style': 'solid', 'border-color': 'white'}
            ), style={'max-width':multi_sel_width,  'width':'250px', 'display': 'inline-block'}))     
    
    
    for i,offset  in enumerate(frames):
        images.append(html.Div(html.Img(id='img'+str(offset), style={'width':'100%', 'border-width': '2px','border-style': 'solid', 'border-color': 'white'}), 
                               style={'overflow':'hidden', 'max-width':max_image_width, 'width':'250px', 'display': 'inline-block'
                                     }, className='zoomlow'))
        labelString=str(offset)
        labels.append(html.Div(html.P(children=labelString, id='lab'+str(offset), style={'color':'black','font-size':'14px','height':'30px', 'background-color':'white','border-width': '2px','border-style': 'solid', 'border-color': 'white'}), 
                               style={'max-height': '30px', 'overflow':'hidden', 'max-width':max_image_width, 'width':'250px', 'display': 'inline-block'}))

        
        sectOptionsButts.append(html.Div(dcc.RadioItems(id='sectButt'+ str(offset),
            options=sectOpts, labelStyle={'display': 'block'},  style={'width':'100%', 'background-color': 'Cornsilk','border-width': '2px','border-style': 'solid', 'border-color': 'white'}
            ), style={'max-width':max_image_width,  'width':'250px', 'display': 'inline-block'}))      

        abnormalOptionsButts.append(html.Div(dcc.RadioItems(id='abButt'+ str(offset),
            options=abnormalOpts, labelStyle={'display': 'block'}, style={'width':'100%', 'background-color': 'MistyRose','border-width': '2px','border-style': 'solid', 'border-color': 'white'}
            ), style={'max-width':max_image_width,  'width':'250px', 'display': 'inline-block'}))   
        
        inflammation.append(html.Div(dcc.Checklist(id='inflammation'+ str(offset),
            options=[{'label': 'inflammation', 'value': 1}], labelStyle={'display': 'block'}, style={'width':'100%', 'background-color': 'RosyBrown','border-width': '2px','border-style': 'solid', 'border-color': 'white'}
            ), style={'max-width':max_image_width,  'width':'250px', 'display': 'inline-block'}))   
        edemous_villi.append(html.Div(dcc.Checklist(id='edemous_villi'+ str(offset),
            options=[{'label': 'edemous_villi', 'value': 1}], labelStyle={'display': 'block'}, style={'width':'100%', 'background-color': 'Tan','border-width': '2px','border-style': 'solid', 'border-color': 'white'}
            ), style={'max-width':max_image_width,  'width':'250px', 'display': 'inline-block'})) 
        bleed.append(html.Div(dcc.Checklist(id='bleed'+ str(offset),
            options=[{'label': 'bleed', 'value': 1}],labelStyle={'display': 'block'}, style={'width':'100%', 'background-color': 'LightCoral','border-width': '2px','border-style': 'solid', 'border-color': 'white'}
            ), style={'max-width':max_image_width,  'width':'250px', 'display': 'inline-block'})) 
        diffuse_bleeding.append(html.Div(dcc.Checklist(id='diffuse'+ str(offset),
            options=[{'label': 'diffuse_bleeding', 'value': 1}],labelStyle={'display': 'block'}, style={'width':'100%', 'background-color': 'DarkSalmon','border-width': '2px','border-style': 'solid', 'border-color': 'white'}
            ), style={'max-width':max_image_width,  'width':'250px', 'display': 'inline-block'})) 
        
        notes.append(html.Div(dcc.Textarea(id='notes'+ str(offset),
            style={'width':'100%', 'background-color': '#fae5d3','border-width': '2px','border-style': 'solid', 'border-color': 'white'}
            ), style={'max-width':max_image_width,  'width':'250px', 'display': 'inline-block'}))   
                
    images=html.Div(images,style={'text-align':'center','margin':'0 auto'} )
    labels=html.Div(labels,style={'text-align':'center','margin':'0 auto'} )
    sectOptionsButts=html.Div(sectOptionsButts,style={'text-align':'center','margin':'0 auto'})
    abnormalOptionsButts=html.Div(abnormalOptionsButts,style={'text-align':'center','margin':'0 auto'})

    inflammation=html.Div(inflammation,style={'text-align':'center','margin':'0 auto'})
    edemous_villi=html.Div(edemous_villi,style={'text-align':'center','margin':'0 auto'})
    diffuse_bleeding=html.Div(diffuse_bleeding,style={'text-align':'center','margin':'0 auto'})
    bleed=html.Div(bleed,style={'text-align':'center','margin':'0 auto'})

    notesFill=html.Div(notes,style={'text-align':'center','margin':'0 auto'})
    #probs=html.Div(probs,style={'text-align':'center','margin':'0 auto'} )

    return html.Div([labels,images,sectOptionsButts,abnormalOptionsButts,inflammation,edemous_villi,bleed, diffuse_bleeding,notesFill])

   
  
    



def buildimages(table):    
    frames=[i['index_'] for i in table]  #[item for item in range(-10,10)] #[-3,-2,-1,0,1,2,3]
    videos=[i['video'] for i in table] 
    images=[]
    labels=[]
    
    for i,offset  in enumerate(frames):
        impath='/project/DSone/jaj4zcf/Videos/'+str(videos[i])+'/'+str(offset)+'.jpg'    ## may need to be updated for final!
        impathdirect='/vids/'+str(videos[i])+'/'+str(offset)+'.jpg'
        ## Only add if file exists
        try:
            if config.live==True:
                images.append(html.Tr(html.Td(html.Div(html.Img(src=impathdirect, style={'width': '100%'}), className='zoom'))))
            else:
                encoded_image = base64.b64encode(open(impath, 'rb').read()).decode("ascii").replace("\n", "")
                images.append(html.Tr(html.Td(html.Div(html.Img(src='data:image/jpg;base64,{}'.format(encoded_image), style={'width': '100%'}), className='zoom'))))
            
            ## add labels
            if offset==0:
                labels.append(html.Td('Selected Frame: ' + str(row)))
            else:
                labels.append(html.Td('Frame: ' + str(offset)))
        except:
            'poo'
   
    images=html.Table(images,style={'width': '99%', 'float':'left'} )
    
    return images




def buildPrevBar(vid, maximum):    
    divs=50
    imsize=str(99/divs)+'%'
    spacing=int(maximum/divs)
    frames=[]
    for i in range(0,divs-1):
        frames.append(i*spacing)
    frames.append(maximum)
    len(frames)
    
    images=[]
    
    for i,offset  in enumerate(frames):
        impath='/project/DSone/jaj4zcf/Videos/'+vid+'/'+str(offset)+'.jpg'    ## may need to be updated for final!
        impathdirect='/vids/'+vid+'/'+str(offset)+'.jpg'
        ## Only add if file exists
        try:
            if config.live==True:
                images.append(html.Td(html.Div(html.Img(src=impathdirect, style={'width': '100%','margin':'0', 'padding':'0'}), className='zoom')))
            else:
                encoded_image = base64.b64encode(open(impath, 'rb').read()).decode("ascii").replace("\n", "")
                images.append(html.Td(html.Div(html.Img(src='data:image/jpg;base64,{}'.format(encoded_image), style={'width': '100%', 'margin':'0', 'padding':'0'}), className='zoom')))
        except:
            'poo'
   
    images=html.Table(html.Tr(images),style={'width': '99%', 'float':'left','cellspacing':'0','cellpadding':'0'} )
    
    return images





vidTable=html.Div(dash_table.DataTable(
        id='vid_table',
        editable=False,
        sort_action = 'native',
        filter_action = 'native',row_selectable='single',
        data=dbf.get_vid_data().to_dict('records'),
        columns=[{"name": i, "id": i} for i in ['video','notes', 'progress']],
        ),style={'width':'auto','overflow':'hidden'})




def build_edit_table(sectOptions, abnormalOptions, ID):
    PAGE_SIZE=10
    COLUMNS=[{"name": i, "id": i} for i in ['index_', 'video', 'tract_section', 'pathology', 'inflammation', 'edemous_villi','bleed','diffuse_bleed','notes']]

    COLUMNS[0].update({ "editable": False})
    COLUMNS[1].update({ "editable": False})
    COLUMNS[2].update({ "presentation": "dropdown"})
    COLUMNS[3].update({ "presentation": "dropdown"})

    table=html.Div(dash_table.DataTable(
        id=ID,
        editable=True,
        sort_action = 'native',
        page_size = PAGE_SIZE,               
        filter_action = 'native',
        row_selectable='single',
        #data=labelsdf[['index','tract_section', 'Pathology', 'Notes']].to_dict('records'),
        columns=COLUMNS,
        #columns[-1][-1]={'name': 'small bowelabNormal', 'id': 'Prob. SB abNorm.'},
        dropdown={
                'tract_section': {
                    'options': [
                        {'label': i, 'value': i}
                        for i in sectOptions
                    ]
                },
                'pathology': {
                    'options': [
                        {'label': i, 'value': i}
                        for i in abnormalOptions
                    ]
                }},
        style_cell={
            'whiteSpace': 'normal',
            'height': '40px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'maxWidth': 0,},
        ),style={'width':'auto','overflow':'hidden'})
    return table


styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

if config.live==True:
    legpath='/vids/legend.PNG'
    leg=html.Img(src=legpath)
else:
    legpath='/project/DSone/jaj4zcf/Videos/legend.PNG'
    encoded_image = base64.b64encode(open(legpath, 'rb').read()).decode("ascii").replace("\n", "")
    leg=html.Img(src='data:image/jpg;base64,{}'.format(encoded_image), style={'height': '40px','width': '80px','float':'right','display':'inline-block', 'vertical-align': 'middle'})
    

    

    
    
application.layout = html.Div(
    [   html.H2('Deep VCE Video Frame Annotation Tool'),
        html.H5('Select a Video From the Video Table to Begin'),
        
        dcc.Tabs([

            
            dcc.Tab(label='Video Selection', children=[html.Div([
                
                html.H4('Current Video Information (select video below to begin):',style={'display':'inline'}),
                
                html.Div([
                html.Div([
                
                html.H5('Video Progress:',style={'display':'inline'}),
                dcc.RadioItems(id='progress',
                    options=[
                        {'label': 'Not Started', 'value': 'Not Started'},
                        {'label': 'In Progress', 'value': 'In Progress'},
                        {'label': 'Complete', 'value': 'complete'}
                    ],
                    value='Not Started',
                    labelStyle={'display': 'inline-block'}
                    )], style={'display':'inline'}),
                html.P('',style={'display':'inline'}),
                html.H5('Video Notes:',style={'display':'inline'}),
                dcc.Textarea(id='videoNotes',draggable='false', title='Area for Notes About Video Overall and Progress', 
                         style={'display':'table','width': '99%','float':'center','margin':'0 auto', 'border-radius': '5px','height': 120}),
                
                html.Button('Save Changes to Video Information', id='save_to_vid_table', n_clicks=0, style={'display':'table'}),
                
            ],style={ 'border-style': 'solid', 'border-width':'thin', 'margin':'5px', "background-color":"LightGray", 'border-radius': '5px'}),
                
                html.H4('Select A Video From the Table Below to View/Edit:'),
                vidTable,
                
                html.Button('Scan For New Videos', id='scan_new_videos', n_clicks=0),
                
                ],style={'margin':'5px', "background-color":"GhostWhite",'border-style': 'thin'} ),
                                                      
              ]),
            
            dcc.Tab(label='Scrub View', children=[
                
                html.Div(children=[
        
                buildTopTableStatic(config.sectOptions,config.abnormalOptions,config.frames),
                #html.Div(id='scrub_table_display', style={'margin':'0 auto', 'width':'100%'}),
                html.Div(id='scrub_display', style={'margin':'auto', 'width':'auto'})], style={'margin':'0 auto', 'width':'100%'}),
                
                
                dcc.Slider(
                    id='scrub_frame',
                    min=0,
                    max=10,
                    step=1,
                    value=0,
                ),
                html.Div(id='scrub_prev'),  

                html.Div([
                html.Button('Previous Set', id='prev_set', n_clicks=0, style={'display':'inline-block', 'vertical-align': 'middle'}),
                html.Button('Previous Frame', id='prev', n_clicks=0, style={'display':'inline-block', 'vertical-align': 'middle'}),
                html.Button('Next Frame', id='next', n_clicks=0, style={'display':'inline-block','vertical-align': 'middle'}),
                html.Button('Next Set', id='next_set', n_clicks=0, style={'display':'inline-block', 'vertical-align': 'middle'}),
                ], style={'display':'inline-block', 'vertical-align': 'middle'}), 
                html.Div([
                html.Button('Set all past X as X', id='set_all', n_clicks=0, style={'float':'right','display':'inline-block', 
                                                                            'vertical-align': 'middle'}),
                ],style={'text-align':'right', 'float':'right','display':'inline-block', 'vertical-align': 'middle'}),
                

                
                
            ]),
            dcc.Tab(label='Table View', children=[
            html.Button('Load Table', id='fill_table', n_clicks=0, style={'display':'inline-block', 'vertical-align': 'middle'}),
            
            html.H5('Use the Data Table Below to explore Results - Click Save Changes to Save Any Changes'),
            html.Button('Save Table Changes', id='save_table', n_clicks=0, style={'display':'inline-block', 'vertical-align': 'middle'}),
            html.P(id='Saved'),
            html.Div([
                html.Div( [
                    html.Div(id='imagecontainertop', style={'height':'80px'}),
                    html.Div(id='imagecontainer', style={'height':'1000px'}),
                ], id='imagecontainerwrap', style={'width':'100px','float':'left','display':'inline-block'}),
                    build_edit_table(config.sectOptions, config.abnormalOptions, ID='table')
                ],
                style={'display':'inline-block', 'width':'100%'}),
            
            html.P('Image Size Slider:'),
            dcc.Slider(
                id='imSize',
                min=100,
                max=512,
                step=1,
                value=100,
            ),  
            html.P('Number of Rows in Table:'),
            dcc.Slider(
                id='num_rows',
                min=3,
                max=20,
                step=1,
                value=10,
            ), 
                
            ]),
            dcc.Tab(label='Review Tool', children=[
            
            html.Button('Load Abnormalities', id='fill_table_review', n_clicks=0, style={'display':'inline-block', 'vertical-align': 'middle'}),
            dcc.RadioItems(
                id='abnormal_options',
                options=[
                    {'label': 'abNormal Pathologies', 'value': 'abnormal_path'},
                    {'label': 'Any Abnormality', 'value': 'any'},
                    {'label': 'Oracle Review', 'value': 'oracle'},
                    {'label': 'Bleeds', 'value': 'bleeds'},
                    {'label': 'Inflammation', 'value': 'inflammation'},
                    {'label': 'Edemous Villi', 'value': 'edemous'}
                ],
                value=['abnormal_path'],
                labelStyle={'display': 'inline-block'}
            ),  
    
                
            html.H5('Use the Data Table Below to explore Results - Click Save Changes to Save Any Changes'),
            html.Button('Save Table Changes', id='save_table_review', n_clicks=0, style={'display':'inline-block', 'vertical-align': 'middle'}),
            html.P(id='Saved_review'),
            html.Div([
                html.Div( [
                    html.Div(id='imagecontainertop_review', style={'height':'80px'}),
                    html.Div(id='imagecontainer_review', style={'height':'1000px'}),
                ], id='imagecontainerwrap_review', style={'width':'100px','float':'left','display':'inline-block'}),
                    build_edit_table(config.sectOptions, config.abnormalOptions, ID='table_review'),
                    html.P(id='test_table_select',children='cow')
                ],
                style={'display':'inline-block', 'width':'100%'}),
            
            html.P('Image Size Slider:'),
            dcc.Slider(
                id='imSize_review',
                min=100,
                max=512,
                step=1,
                value=100,
            ),  
            html.P('Number of Rows in Table:'),
            dcc.Slider(
                id='num_rows_review',
                min=3,
                max=20,
                step=1,
                value=10,
            ), 
                
            ]),
            
            
            
            
        ]),
         

        ## may need to be updated for final!
        

    
     
         
    
    

    html.P(id='table_name',style={'display': 'none'} ),
        
    html.Div(id='table-var', style={'display': 'none'}),  #where to store the table values.
    html.Div(id='offset-var', children=[0], style={'display': 'none'}),  #where to store the offset from table to index. 

    html.Div(id='frame-var',children=[0], style={'display': 'none'}),  #store the current frame 
    html.Div(id='prev-click',children=[0], style={'display': 'none'}),
    
    html.Div(id='num-prev',children=[0], style={'display': 'none'}),
    html.Div(id='num-next',children=[0], style={'display': 'none'}),
    
    #To Prevent Update On Startups
    html.P('',id='placeholder', style={'display': 'none'}),
     
    html.P('',id='placeholder2', style={'display': 'none'}),
    html.P('',id='placeholder3', style={'display': 'none'}),
    html.P('',id='set_all_var', style={'display': 'none'}),
    html.P('',id='last_scrub_but', style={'display': 'none'}),
     
    html.P(id='test'),
    html.Div([Keyboard(id="keyboard"), html.Div(id="output")])
                      ], style={'max-width':'100%'})


## ALL INPUTS TO TABLE

## Select Video in Video Table by Selecting Row in Review tool.
@application.callback(
    Output('vid_table', 'selected_rows'),
    [Input('table_review', 'derived_viewport_selected_rows')], [State('table_review', 'derived_viewport_data'),State('vid_table', 'data')])   # Input('table', "derived_virtual_selected_rows")
def display_click_data(selected_rows,data,vid_table_data):
    try:
        video = str(data[int(selected_rows[0])]['video'])
        vidTable=pd.DataFrame(vid_table_data)
        return np.where(vidTable.video==video)[0] #indi = str(data[int(selected_rows[0])]['index_'])
        
    except:
        raise dash.exceptions.PreventUpdate


        
@application.callback(
    Output('test_table_select', 'children'),
    [Input('table_review', 'derived_viewport_selected_rows')], [State('table_review', 'derived_viewport_data'),State('vid_table', 'data')])   # Input('table', "derived_virtual_selected_rows")
def display_click_data(selected_rows,data,vid_table_data):
    try:
        return str(data[int(selected_rows[0])]['index_'])
        
    except:
        raise dash.exceptions.PreventUpdate        

        
@application.callback(Output("output", "children"), [Input("keyboard", "keydown")])
def keydown(event):
    return json.dumps(event)

@application.callback(Output("placeholder3", "children"), [Input("scan_new_videos", "n_clicks")], [State('placeholder3','children')])
def scan_for_new(n_clicks,ignit):
    if ignit=='initialized':
        dbf.scan_for_new_videos()
    return 'initialized'



@application.callback(
     [Output('table_name', 'children'),Output('progress','value'), Output('videoNotes','value')] ,
    [Input('vid_table', 'selected_rows'), Input('vid_table', 'data')]
    )
def update_table_name(selected_rows, data):

    if selected_rows is None:
        raise dash.exceptions.PreventUpdate
    else:

        vname = str(data[int(selected_rows[0])]['video'])
        row=dbf.get_vid_data_row(vname)
        return vname, row[2], row[1]




# Builds Images on Table View
@application.callback(
    Output('imagecontainer', 'children'),
    [Input('table', 'derived_viewport_data') ])
def update_image_div(table):

    try:
        images = buildimages(table)
        return images
    except:
        return None

    
    
    
    
# Builds Images on Table View
@application.callback(
    Output('imagecontainer_review', 'children'),
    [Input('table_review', 'derived_viewport_data') ])
def update_image_div(table):

    try:
        images = buildimages(table)
        return images
    except:
        return None

@application.callback(Output('Saved','children'),
            [Input('save_table','n_clicks')],
                     [State('table','data'), State('table_name','children')])
def save_table(n_clicks, data,vname):   
    
    data=pd.DataFrame(data)
    labelsdf=dbf.get_video_df(vname)
    data=data[['index_', 'tract_section', 'pathology','inflammation','edemous_villi','bleed','diffuse_bleed','notes']]
    labelsdf=labelsdf[['index_', 'tract_section', 'pathology','inflammation','edemous_villi','bleed','diffuse_bleed','notes']]
    ans=[any(row[1]) for row in (~data.isin(labelsdf)).iterrows()]
    data=data[ans]
    j=0
    for i,row in data.iterrows():
        j=j+1
        dbf.update_row(vname,(row['tract_section'], row['pathology'],
                       row['notes'], row['inflammation'],row['edemous_villi'],row['bleed'],row['diffuse_bleed']),row['index_'])
    return 'Saved ' + str(j) + ' rows at '+ str(datetime.datetime.now())
   #except:
   #     raise dash.exceptions.PreventUpdate
    
    
 
    
@application.callback(
    [Output('table', 'style_data'), Output('imagecontainerwrap', 'style')],
    [Input('imSize', 'value')])   # Input('table', "derived_virtual_selected_rows")
def tablepicsize(value):
    
    cellwidth=str(value)+'px'
    style_cell={'whiteSpace': 'normal', 'height': cellwidth} 
    style_wrap={'width':cellwidth,'float':'left','display':'inline-block'}
    
    return style_cell, style_wrap


@application.callback(
    [Output('table_review', 'style_data'), Output('imagecontainerwrap_review', 'style')],
    [Input('imSize_review', 'value')])   # Input('table', "derived_virtual_selected_rows")
def tablepicsize(value):
    
    cellwidth=str(value)+'px'
    style_cell={'whiteSpace': 'normal', 'height': cellwidth} 
    style_wrap={'width':cellwidth,'float':'left','display':'inline-block'}
    
    return style_cell, style_wrap
    

####
####
@application.callback(
    Output('set_all_var', 'children'),
    [Input('set_all','n_clicks')],
    [State('sectButt0', 'value'), State('scrub_frame', 'value'),State('table_name','children')])
     # Input('table', "derived_virtual_selected_rows")
def update_rest_tract(n_clicks,sect,frame,vname):
    try:
        ctx = dash.callback_context
        if not ctx.triggered:
            button_id = 'No clicks yet'
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
       
        if button_id == 'set_all':
            dbf.set_rest_tract(vname,sect,frame)
            return 'Complete'
        else:
            raise dash.exceptions.PreventUpdate
    except:
        raise dash.exceptions.PreventUpdate

####
@application.callback(
    Output('set_all', 'children'),
    [Input('sectButt0', 'value'), Input('scrub_frame', 'value')])
     # Input('table', "derived_virtual_selected_rows")
def update_rest_tract(sect,frame):
    try:
        val='Set all past frame:'+str(frame)+' as ' + sect
        return val
    except:
        raise dash.exceptions.PreventUpdate
    
####
@application.callback(
    [Output('placeholder2', 'children'),Output('vid_table', 'data')],
    [Input('save_to_vid_table', 'n_clicks')],
    [State('placeholder2', 'children'), State('table_name','children'), State('progress','value'), State('videoNotes','value')])   # Input('table', "derived_virtual_selected_rows")
def update_vid_table(n_clicks,initialized, vname, progress, notes):

    if initialized == 'initialized':

        dbf.update_video_row(vname, progress, notes)
    data=dbf.get_vid_data().to_dict('records')

    return 'initialized', data




## Change table selection based on current frame and using offset (table does not start at 0)
@application.callback(
    [Output('scrub_frame', 'value'), Output('last_scrub_but','children')],
    [Input('next', 'n_clicks'),Input('prev', 'n_clicks'),
     Input('next_set', 'n_clicks'),Input('prev_set', 'n_clicks'),
     Input('scrub_frame', 'min'), Input('scrub_frame', 'max'), Input("keyboard", "keydown"),
     Input('table_review', 'derived_viewport_selected_rows')],
    [State('scrub_frame', 'value'),State('table_review', 'derived_viewport_data'),State('last_scrub_but','children')])
def diplay_table(next_n_clicks,prev_n_clicks, next_set, prev_set, min_val, max_val, keyboard, selected_rows,frame,data,last_scrub_but):

    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
        comp_id = 'No Value'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        comp_id = ctx.triggered[0]['prop_id'].split('.')[1]

    try:
        keyboard=dict(keyboard)
    except:
        keyboard={'key':'poo'}
              
    if button_id == 'next':
        try:    
            val=frame+1
            if val > max_val:
                val=max_val
            return val, button_id #[int(test['points'][0]['x'])]
        except:
            raise dash.exceptions.PreventUpdate
            
    elif button_id == 'prev':
        try:    
            val=frame-1
            if val<0:
                val=0
            return val, button_id #[int(test['points'][0]['x'])]
        except:
            raise dash.exceptions.PreventUpdate
    
    elif button_id == 'next_set':
        try:    
            val=frame+int(config.n_images)
            if val > max_val:
                val=max_val
            return val, button_id #[int(test['points'][0]['x'])]
        except:
            raise dash.exceptions.PreventUpdate
   
    elif button_id == 'prev_set':
        try:    
            val=frame-int(config.n_images)
            if val<0:
                val=0
            return val, button_id #[int(test['points'][0]['x'])]
        except:
            raise dash.exceptions.PreventUpdate
   
    elif button_id == 'scrub_frame':
        try:    
            if last_scrub_but!=table_review:
                return int(min_val), button_id 
            else:
                raise dash.exceptions.PreventUpdate
        except:
            raise dash.exceptions.PreventUpdate
    
    elif button_id == 'keyboard' and keyboard['key'] in ['ArrowRight']:
        try:    
            val=frame+int(config.n_images)
            if val > max_val:
                val=max_val
            return val, button_id #[int(test['points'][0]['x'])]
        except:
            raise dash.exceptions.PreventUpdate 
    elif button_id == 'keyboard' and keyboard['key'] in ['ArrowLeft']:
        try:    
            val=frame-int(config.n_images)
            if val<0:
                val=0
            return val, button_id #[int(test['points'][0]['x'])]
        except:
            raise dash.exceptions.PreventUpdate
    elif button_id == 'table_review':
        return int(data[int(selected_rows[0])]['index_']), button_id
    else:
        raise dash.exceptions.PreventUpdate
    



@application.callback(
    Output('table', 'page_size'),
    [Input('num_rows', 'value')])   # Input('table', "derived_virtual_selected_rows")
def pages(value):
    return value


@application.callback(
    Output('table_review', 'page_size'),
    [Input('num_rows_review', 'value')])   # Input('table', "derived_virtual_selected_rows")
def pages(value):
    return value

    
    
    


### Callback must

sectInputs=[State('sectButt'+str(offset), 'value') for offset in config.frames]
abInputs=[State('abButt'+str(offset), 'value') for offset in config.frames]
notesInputs=[State('notes'+str(offset), 'value') for offset in config.frames]

inflammationInputs=[State('inflammation'+str(offset), 'value') for offset in config.frames]
edemous_villiInputs=[State('edemous_villi'+str(offset), 'value') for offset in config.frames]
diffuse_bleedInputs=[State('diffuse'+str(offset), 'value') for offset in config.frames]
bleedInputs=[State('bleed'+str(offset), 'value') for offset in config.frames]

allinputs=sectInputs+abInputs+notesInputs+inflammationInputs+edemous_villiInputs+bleedInputs+diffuse_bleedInputs

@application.callback(Output('placeholder','children'),
                     [Input('next', 'n_clicks'),Input('prev', 'n_clicks'),
                     Input('next_set', 'n_clicks'),Input('prev_set', 'n_clicks'), Input('keyboard', 'keydown')],
                     [State('table_name', 'children'), State('scrub_frame', 'value'), State('placeholder', 'children')] + allinputs)
def saveScrubInputs(next_c, prev_c, next_s_c, prev_s_c, keyboard, vname, frame, initialized, *arg):
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
        comp_id = 'No Value'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        comp_id = ctx.triggered[0]['prop_id'].split('.')[1]
    
    try:
        keyboard=dict(keyboard)
    except:
        keyboard={'key':'poo'}
    
    
    
    if initialized == 'initialized':
        nrows=int(len(arg)/7)
        framenums=[f+frame for f in config.frames]
        sects=arg[0:nrows]
        pathos=arg[nrows:2*nrows]
        notes=arg[2*nrows:3*nrows]
        inflammation=arg[3*nrows:4*nrows]
        edemous_villi=arg[4*nrows:5*nrows]
        bleed=arg[5*nrows:6*nrows]
        diffuse=arg[6*nrows:7*nrows]
        values=[]
        for i in range(0,nrows):
            if len(inflammation[i])>0:
                inf=inflammation[i][-1]
            else:
                inf=0
            if len(edemous_villi[i])>0:
                edem=edemous_villi[i][-1]
            else:
                edem=0
            if len(bleed[i])>0:
                blee=bleed[i][-1]
            else:
                blee=0    
            if len(diffuse[i])>0:
                diff=diffuse[i][-1]
            else:
                diff=0          
            values.append([framenums[i],sects[i],pathos[i],notes[i],inf,edem,blee,diff])    
        #values=values[frame]
        values=[row for row in values if row[0] >= 0 ]
        dbf.update_rows(vname,values)  
    
    return 'initialized'




sectOuts=[Output('sectButt'+str(offset), 'value') for offset in config.frames]
abOuts=[Output('abButt'+str(offset), 'value') for offset in config.frames]
notesOuts=[Output('notes'+str(offset), 'value') for offset in config.frames]

inflammationOuts=[Output('inflammation'+str(offset), 'value') for offset in config.frames]
edemous_villiOuts=[Output('edemous_villi'+str(offset), 'value') for offset in config.frames]
diffuse_bleedOuts=[Output('diffuse'+str(offset), 'value') for offset in config.frames]
bleedOuts=[Output('bleed'+str(offset), 'value') for offset in config.frames]

sectStates=[State('sectButt'+str(offset), 'value') for offset in config.frames]
abStates=[State('abButt'+str(offset), 'value') for offset in config.frames]
notesStates=[State('notes'+str(offset), 'value') for offset in config.frames]

inflammationStates=[State('inflammation'+str(offset), 'value') for offset in config.frames]
edemous_villiStates=[State('edemous_villi'+str(offset), 'value') for offset in config.frames]
diffuse_bleedStates=[State('diffuse'+str(offset), 'value') for offset in config.frames]
bleedStates=[State('bleed'+str(offset), 'value') for offset in config.frames]

@application.callback(sectOuts+abOuts+notesOuts+inflammationOuts+edemous_villiOuts+bleedOuts+diffuse_bleedOuts,        
                     [Input("clone", "n_clicks"), Input('table_name', 'children'), Input('scrub_frame', 'value'), Input('set_all_var', 'children'),
                      Input('abButt_multi', 'value'),Input('notes_multi', 'value'),Input('inflammation_multi', 'value'),Input('edemous_villi_multi', 'value'),Input('diffuse_multi', 'value'),Input('bleed_multi', 'value')],
                     sectStates+abStates+notesStates+inflammationStates+edemous_villiStates+bleedStates+diffuse_bleedStates)
def popvalues(clone, vname, frame, set_all, abButt_multi, notes_multi,inflammation_multi,edemous_villi_multi,diffuse_multi,bleed_multi,*arg):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
        comp_id = 'No Value'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        comp_id = ctx.triggered[0]['prop_id'].split('.')[1]
    
    nrows=int(len(arg)/7)
    framenums=[f+frame for f in config.frames]
    sects=arg[0:nrows]
    pathos=arg[nrows:2*nrows]
    notes=arg[2*nrows:3*nrows]
    inflams=arg[3*nrows:4*nrows]
    edems=arg[4*nrows:5*nrows]
    bleeds=arg[5*nrows:6*nrows]
    diffuses=arg[6*nrows:7*nrows]
   
    if button_id not in ['abButt_multi','notes_multi','edemous_villi_multi','diffuse_multi','inflammation_multi','bleed_multi', 'clone']:
        try:
            values=dbf.read_set(vname, frame, config.frames)
            blanks=len(config.frames)-len(values)

            sects=[]
            pathos=[]
            notes=[]
            inflams=[]
            edems=[]
            diffuses=[]
            bleeds=[]
            
            for i in range(0,blanks):
                sects.append('')
                pathos.append('')
                notes.append('')
                inflams.append('')
                edems.append('')
                bleeds.append('')
                diffuses.append('')

            for row in values:
                sects.append(row[0])
                pathos.append(row[1])
                notes.append(row[2])
                inflams.append([row[3]])
                edems.append([row[4]])
                bleeds.append([row[5]])
                diffuses.append([row[6]])

            return sects + pathos + notes + inflams + edems + bleeds + diffuses
        except:
            raise dash.exceptions.PreventUpdate
    elif button_id=='abButt_multi':
        return list(sects) + [abButt_multi for k in range(0,nrows)] + list(notes) + list(inflams) + list(edems) + list(bleeds) + list(diffuses)
    elif button_id=='notes_multi':
        return list(sects) + list(pathos) + [notes_multi for k in range(0,nrows)] + list(inflams) + list(edems) + list(bleeds) +list(diffuses)
    elif button_id=='inflammation_multi':
        return list(sects) + list(pathos) +  list(notes) + [inflammation_multi for k in range(0,nrows)] + list(edems) +list(bleeds) + list(diffuses)
    elif button_id=='edemous_villi_multi':
        return list(sects) + list(pathos) + list(notes)  + list(inflams) + [edemous_villi_multi for k in range(0,nrows)] +list(bleeds) + list(diffuses)
    elif button_id=='bleed_multi':
        return list(sects) + list(pathos) + list(notes) + list(inflams) + list(edems) + [bleed_multi for k in range(0,nrows)] + list(diffuses) 
    elif button_id=='diffuse_multi':
        return list(sects) + list(pathos) + list(notes) + list(inflams) + list(edems) + list(bleeds) + [diffuse_multi for k in range(0,nrows)]
    elif button_id=='clone':
        return list(sects) + [abButt_multi for k in range(0,nrows)] + [notes_multi for k in range(0,nrows)] + [inflammation_multi for k in range(0,nrows)] + [edemous_villi_multi for k in range(0,nrows)] + [bleed_multi for k in range(0,nrows)]  + [diffuse_multi for k in range(0,nrows)]
### Select Video Table From Database - update 


@application.callback([Output('scrub_frame', 'max')],        
                     [Input('table_name', 'children')])
def set_scrub_bar(vname):
    try:
        max_frame=dbf.get_max_frame(vname)
        return[max_frame]
    except:
        raise dash.exceptions.PreventUpdate

        
@application.callback([Output('scrub_prev','children')],        
                     [Input('scrub_frame', 'max')],
                     [State('table_name', 'children')])
def buildscrubprev(maximum, vid):
    try:
        prev=buildPrevBar(vid, maximum)   
        return [prev]
    except:
        raise dash.exceptions.PreventUpdate
        
         


### Load table from database
@application.callback([Output('table', 'data')],        
                     [Input('fill_table', 'n_clicks'), Input('table_name', 'children')])
def return_table_data(fill_table, vname):
    try:
        ctx = dash.callback_context
        if not ctx.triggered:
            button_id = 'No clicks yet'
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        ### only update if videoselect changed.....
        if button_id=='fill_table':
            labelsdf=dbf.get_video_df(vname)

        else:
            raise dash.exceptions.PreventUpdate

        return [labelsdf.to_dict('records')]
    except:
        raise dash.exceptions.PreventUpdate

############
### Load Table for Review and Save Table For Review
############

# dictionary shared between two callbacks below
condition_dict = {        "abnormal_path": 'pathology !="normal"',
                          "any": 'pathology !="normal" OR inflammation>0 OR edemous_villi>0 OR bleed>0',
                          "bleeds": 'bleed>0',
                          "edemous": 'edemous_villi>0',
                          "inflammation": 'inflammation>0'}

############ Load Abnormals Table

@application.callback([Output('table_review', 'data')],        
                     [Input('fill_table_review', 'n_clicks')], [State('abnormal_options','value')])
def return_table_data(fill_table,abnormal):
    # add special case for oracle later
    condition = condition_dict[abnormal]
    tables=dbf.findnames()
    tables=[x for x in tables if x!='prog_table']
    labelsdf=pd.DataFrame(dbf.get_anoms(tables, condition), columns=['index_','video', 'tract_section', 'pathology', 'inflammation', 'edemous_villi', 'bleed', 'diffuse_bleed','notes'])
    return [labelsdf.to_dict('records')]


############ Save Changes to Abnormals Table

@application.callback(Output('Saved_review','children'),
            [Input('save_table_review','n_clicks')],
                     [State('table_review','data'), State('abnormal_options','value')])
def save_table(n_clicks, data,abnormal):   
    data=pd.DataFrame(data)
    condition = condition_dict[abnormal]
    tables=dbf.findnames()
    tables=[x for x in tables if x!='prog_table']
    labelsdf=pd.DataFrame(dbf.get_anoms(tables, condition), columns=['index_','video', 'tract_section', 'pathology', 'inflammation', 'edemous_villi', 'bleed', 'diffuse_bleed','notes'])
    data=data[['index_','video', 'tract_section', 'pathology','inflammation','edemous_villi','bleed','diffuse_bleed','notes']]
    labelsdf=labelsdf[['index_', 'video','tract_section', 'pathology','inflammation','edemous_villi','bleed','diffuse_bleed','notes']]
    ans=[any(row[1]) for row in (~data.isin(labelsdf)).iterrows()]
    
    
    data=data[ans]
    j=0
    rows_to_update=[]
    for i,row in data.iterrows():
        j=j+1
        rows_to_update.append([row['video'], row['tract_section'], row['pathology'],
                       row['notes'], row['inflammation'],row['edemous_villi'],row['bleed'],row['diffuse_bleed'],row['index_']])
    
    dbf.update_multi_row(rows_to_update)
    return 'Saved ' + str(data) + ' rows at '+ str(datetime.datetime.now())  
   

### Update Images on Annotation Scrub Table
@application.callback(
    [Output('img'+str(offset), 'src') for offset in config.frames],
    [Input('scrub_frame', 'value'), Input('table_name', 'children')])
def update_nscrub_images(scrub_frame, vid):
    try:
        if vid[-2].isnumeric():
            vid=vid[-2:]
        else:
            vid=vid[-1]

        imgsrcs=[]
        for offset in config.frames:
            if offset+scrub_frame>=0:
                imgsrcs.append(getImage(vid, scrub_frame+offset))
            else:
                imgsrcs.append(None)

        return imgsrcs
    except:
        raise dash.exceptions.PreventUpdate
### Update Image Indexes on Annotation Scrub Table
@application.callback(
    [Output('lab'+str(offset), 'children') for offset in config.frames],
    [Input('scrub_frame', 'value')])
def update_nscrub_(scrub_frame):
    try:
        framenums=[]
        for offset in config.frames:
            framenums.append(scrub_frame+offset)
        return framenums
    except:
        raise dash.exceptions.PreventUpdate



@application.server.route('/vids/<path:path>')
def send_jss(path):
    pos=-(path[::-1].find('/'))
    filename=path[pos::]
    path=path[0:pos]
    return send_from_directory(filespath+path, filename)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
