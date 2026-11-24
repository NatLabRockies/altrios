# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 09:59:15 2021

@author: groscoe2
"""
print('Initializing')
# import matplotlib.pyplot as plt
import json
import plotly.graph_objects as go
# import plotly.express as px
import os
from inspect import getsourcefile
import random
import progressbar
from time import sleep

warmupLength = 24 #hours
cooldownLength = 24  # hours
numDays = 9 #full simulation length
directory = ''
# directory = 'C:/Users/MMP-S/Downloads/' # Results Directory
is_directory_full_path = False
plotName = 'GraphTestTrack.html' #name of plot file
networkOverride = False # 'C:/Users/MMP-S/Downloads/network.json' #if ussing a different network file than one in results directory
checkSubfolders = True #walk through subfolders looking for results
overwriteExisting = False #overwrite existing plots
colors = ['#000000', '#FF0000', '#FF9900','#0000FF','#FF00FF', '#14B460', '#969696', '#993366', '#00DFDA',   ] # line graph colors, if unspecified, will use default plotly colors
xLabel = 'Time (Days)'
def yLabel(folder):
    return ('Distance (Miles)')
narrowWidth = 2 #narrow width of lines
wideWidth = 5 #wide width of lines





fileDirect = os.path.abspath(getsourcefile(lambda: 0))
path, file = os.path.split(fileDirect)
# directories = os.listdir(directory)
if is_directory_full_path:
    full_path = directory
else:
    full_path = os.path.join(path, directory)

def combine_by_entry(listDict, variable, x = 'x', y = 'y', sortingKey = None):
    if sortingKey == None:
        sortingKey = lambda x:x[variable]
    newListDict = []
    variableList = []
    for entry in listDict:
        if entry[variable] not in variableList:
            variableList.append(entry[variable])
            newListDict.append({
                x:[],
                y:[],
                variable:entry[variable]})
            dictIndex = -1
        else:
            dictIndex = variableList.index(entry[variable])
        newListDict[dictIndex][x].append(None)
        newListDict[dictIndex][y].append(None)
        if type(entry[x]) == list:
            newListDict[dictIndex][x].extend(entry[x])
        else:
            newListDict[dictIndex][x].append(entry[x])
            
        if type(entry[y]) == list:
            newListDict[dictIndex][y].extend(entry[y])
        else:
            newListDict[dictIndex][y].append(entry[y])
    newListDict.sort(key = sortingKey)
    return newListDict

def sort_by_length(inputList, variable = 'distance'):
    length = 0
    for x in range(len(inputList)):
        if x == 0:
            continue
        elif inputList[variable][x] == None or inputList[variable][x-1] == None:
            continue
        else:
            length += abs(inputList[variable][x] - inputList[variable][x-1])
    if length != 0:
        return 1/length
    else:
        return float('inf')



scenarios = []
if networkOverride != False:
    print('Loading Network Override File')
    with open(networkOverride) as networkFile:
        networkData = json.load(networkFile)['links']
    linkTracks = ['None'] * len(networkData)
    for link in range(len(networkData)):
        if 'linkName' in networkData[link].keys():
            if 'track' in networkData[link]['linkName'].keys():
                linkTracks[link] = networkData[link]['linkName']['track']
    del networkData
print('Compiling File and Folder List')
if checkSubfolders:
    allFolders = (list(os.walk(full_path)))
else:
    allFolders = (full_path,'',os.listdir(full_path))
    
filesExist = False
if overwriteExisting:
    print('Removing Previous Plots')
    for x  in range(len(allFolders)):
        folder, subfolders, files = allFolders[x]
        if 'dispOccupancy.json' in files and 'dispSummary.json' in files and plotName in files and (networkOverride != False + ('network.json' in files)):
            os.remove(folder+"/"+plotName)
            allFolders[x][2].remove(plotName)
    # if filesExist:
    #     if checkSubfolders:
    #         allFolders = (list(os.walk(psm.get_full_path(directory,path))))
    #     else:
    #         allFolders = (psm.get_full_path(directory,path),'',os.listdir(psm.get_full_path(directory,path)))
            
random.shuffle(allFolders)
completeFolders = []

narrowDict = []
normalDict = []
wideDict = []
if colors ==None:
    narrrowDict = {'width' : narrowWidth}
    normalDict = {'width':float((narrowWidth + wideWidth) / 2)}
    wideDict = {'width':wideWidth}
else:
    for color in colors:
        narrowDict.append({
            'width':narrowWidth,
            'color':color})
        normalDict.append({
            'width':float((narrowWidth + wideWidth) / 2),
            'color':color})
        wideDict.append({
            'width':wideWidth,
            'color':color})

bar = progressbar.ProgressBar(max_value=15)
print('Generating Graphs')
bar.start()

unwrittenFiles = []

for x  in range(len(allFolders)):
    bar.update(x/len(allFolders) * 15)

    folder, subfolders, files = allFolders[x]
    # print(folder)
    if folder in completeFolders:
        # print('Complete Folder')
        continue
    else:
        completeFolders.append(folder)
    if plotName in files:
        # print('Complete')
        continue
    elif 'dispOccupancy.json' in files and 'dispSummary.json' in files and (networkOverride != False + ('network.json' in files)):
        if os.path.isfile(folder+'/'+plotName):
            continue
        else:
            with open(folder+'/'+plotName,'w') as tempFile:
                tempFile.write(os.environ['COMPUTERNAME'])
        # print('folder')
        if networkOverride == False:
            with open(folder+'/network.json') as networkFile:
                networkData = json.load(networkFile)['links']
                linkTracks = ['None'] * len(networkData)
                for link in range(len(networkData)):
                    if 'linkName' in networkData[link].keys():
                        if 'track' in networkData[link]['linkName'].keys():
                            linkTracks[link] = networkData[link]['linkName']['track']
                del networkData
        
        with open(folder+"/dispOccupancy.json") as f:
            dispResult = json.load(f)['dispResult']
            

        
        # colorOrder = ['#0000FF','#FF0000', '#FF9900', '#14B460', '#00DFDA', '#993366','#969696', '#FF00FF']
        typeOrder = ['solid']
        trackList = []
        occupancyList = []
        startLocations = []
        for linkIdx in range(len(dispResult)):
            for entry in dispResult[linkIdx]['trainTimes']:
                while len(startLocations) < entry['trainIdx']+1:
                    startLocations.append((float('inf'),None))
                occupancyList.append(
                    {'time':[entry['timeArrStart']/86400-warmupLength / 24,entry['timeArrEnd']/86400 - warmupLength / 24],
                      'distance':[dispResult[linkIdx]['offsetStart'] / 1609.344,dispResult[linkIdx]['offsetEnd']/1609.344],
                      'trainIdx':entry['trainIdx'],
                      'Primary Route': dispResult[linkIdx]['trackType'],
                      'Track Name': linkTracks[linkIdx]}
                    )
                # print(trackList)
                if startLocations[entry['trainIdx']][0] > entry['timeArrStart']/86400:
                    startLocations[entry['trainIdx']] = (entry['timeArrStart']/86400,dispResult[linkIdx]['offsetStart'] / 1609.344)
                if startLocations[entry['trainIdx']][0] > entry['timeArrEnd']/86400:
                    startLocations[entry['trainIdx']] = (entry['timeArrEnd']/86400,dispResult[linkIdx]['offsetEnd'] / 1609.344)
        with open(folder+"/"+'dispSummary.json') as f:
            dispSummary = json.load(f)['dispSummary']
        for trainIdx in range(len(dispSummary)):
            train = dispSummary[trainIdx]
            if train['timeSchedStart'] != train['timeActualStart']:
                occupancyList.append(
                    {'time':[train['timeSchedStart']/86400 - warmupLength / 24,train['timeActualStart']/86400 - warmupLength / 24],
                      'distance':[startLocations[trainIdx][1],startLocations[trainIdx][1]],
                      'trainIdx':trainIdx,
                      'Primary Route':-1,
                      'Track Name':'Off Network'})
        # trackList.sort(key = lambda x:x[2])
        # minLocation = min([
        #     min(trackList,key = lambda x:x[1][0])[1][0],
        #     min(trackList,key = lambda x:x[1][1])[1][1],])
        # maxLocation = min([
        #     max(trackList,key = lambda x:x[1][0])[1][0],
        #     max(trackList,key = lambda x:x[1][1])[1][1],])
        filterButtons=[]
        buttons = []
        sizeButtons = []
        trackList = combine_by_entry(occupancyList,'Track Name',x = 'time',y = 'distance',sortingKey = sort_by_length)
        trainList = combine_by_entry(occupancyList,'trainIdx',x = 'time',y = 'distance')
        
        
        fig = go.Figure()
        traces=[]
        trackTraces = []
        trainTraces = []
        index = 0
        for line in trackList:
            trackTraces.append(go.Scatter(
                            x = line['time'],
                            y = line['distance'],
                            name = 'Track: '+line['Track Name'],
                            line = dict(
                            color = colors[index % len(colors)],
                            width = narrowWidth
                            ),
                            visible = True))
            index += 1
            
        index = 0
        for line in trainList:
            trainTraces.append(go.Scatter(
                            x = line['time'],
                            y = line['distance'],
                            name = 'Train Idx: '+str(line['trainIdx']),
                            line = dict(
                            color = colors[index % len(colors)],
                            width = narrowWidth
                            ),
                            visible = False))
            index += 1
        traces.extend(trackTraces)
        traces.extend(trainTraces)
        filterButtons.append(dict(
            method = 'update',
              label = 'Track Type',
              visible = True,
              args = [{'visible':[True]*len(trackTraces)+[False]*len(trainTraces)},],
              # args2 = [{'visible':[False]*len(trackTraces)+[True]*len(trainTraces)}]
              ))
        filterButtons.append(dict(
            method = 'update',
              label = 'Train Index',
              visible = True,
              args = [{'visible':[False]*len(trackTraces)+[True]*len(trainTraces)}],
              # args2 = []##
              ))
        sizeButtons.append(dict(
            method = 'update',
              label = 'Narrow Lines',
              visible = True,
              # args3 = [{'line': {'width':10}}],
              args = [{'line' : narrowDict}],
              ))
        sizeButtons.append(dict(
            method = 'update',
              label = 'Normal Lines',
              visible = True,
              # args3 = [{'line': {'width':10}}],
              args = [{'line' : normalDict}],
              ))
        sizeButtons.append(dict(
            method = 'update',
              label = 'Wide Lines',
              visible = True,
              # args3 = [{'line': {'width':10}}],
              args = [{'line' : wideDict}],
              ))
        buttons.append(dict(
            method = 'relayout',
            visible = True,
            label = 'Legend',
            args = [{'showlegend':True
                # overlap = False
            }],
            args2 = [{'showlegend' : False
            }]
            ))
        # buttons.append(dict(
        #     method = 'update',
        #       label = 'Train',
        #       visible = True,
        #       args = [{'visible':True},list(range(len(trackTraces),len(trackTraces)+len(trainTraces)))],
        #       args2 = [{'visible':False},list(range(len(trackTraces),len(trackTraces)+len(trainTraces)))]
        #       ))
        for trace in traces:
            fig.add_trace(trace)
        fig.update_layout(
            plot_bgcolor="white",
            updatemenus=[
                dict(
                    type = "buttons",
                    direction = "left",
                    buttons=filterButtons,
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                ),
                dict(
                type = "buttons",
                # direction = "down",
                buttons=buttons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.5,
                xanchor="center",
                y=1.1,
                yanchor="top"
                    ),
                dict(
                type = "buttons",
                direction = "right",
                buttons=sizeButtons,
                pad={"r": 10, "t": 10},
                showactive=True,
                x=1,
                xanchor="right",
                y=1.1,
                yanchor="top"
                    )
            ]
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='grey')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='grey')
        fig.update_layout(
            xaxis_title=xLabel,
            yaxis_title=yLabel(folder),
            font = dict(family = 'Arial',
                        size = 16,
                        color = 'black'),
            xaxis_title_font =dict(
                size=20,
                family = "Arial Black"
            ),
            yaxis_title_font =dict(
                size=20,
                family = "Arial Black"
            ),
            xaxis_tickfont= dict(
                family = 'Arial Black',
                size = 18),
            yaxis_tickfont= dict(
                family = 'Arial Black',
                size = 18)
        )
        # fig.update_layout(legend=dict(
        #     yanchor="top",
        #     y=1,
        #     xanchor="left",
        #     x=0,
        #     # overlap = False
        # ))
        try:
            fig.write_html(folder+"/"+plotName)
        except IOError as e:
            unwrittenFiles.append((e, folder+"/"+plotName, go.Figure(fig)))
        # print(folder+"/"+plotName)
        # for occupancy in trackList:
        #     plt.plot(occupancy[0],occupancy[1],
        #              color=colorOrder[occupancy[2] % len(colorOrder)],
        #              linestyle = typeOrder[occupancy[2] % len(typeOrder)],
        #              linewidth = 1)
        # # plt.plot([0,0],[minLocation, maxLocation],color = 'black',linestyle = (0, (1, 10)))
        # # plt.plot([numDays - cooldownLength / 24 - warmupLength / 24,
        # #           numDays - cooldownLength / 24 - warmupLength / 24],
        # #          [minLocation,maxLocation],color = 'black',linestyle = (0, (1, 10)))
        # # plt.plot([])
        # plt.ylabel('Miles from Savannah')
        # plt.xlabel('Day')
        # plt.savefig('TrainTest.jpeg', dpi = 1000,  bbox_inches='tight')
        # plt.clf()
        
        # trackList.sort(key = lambda x:x[3])
        # for occupancy in trackList:
        #     plt.plot(occupancy[0],occupancy[1],
        #              color=colorOrder[occupancy[3] % len(colorOrder)],
        #              linestyle = typeOrder[occupancy[2] % len(typeOrder)],
        #              linewidth = 1.5)
        # # plt.plot([0,0],[minLocation,maxLocation],color = 'black',linestyle = (0, (1, 10)))
        # # plt.plot([numDays - cooldownLength / 24 - warmupLength / 24,
        # #           numDays - cooldownLength / 24 - warmupLength / 24],
        # #          [minLocation,maxLocation],color= 'black',linestyle = (0, (1, 10)))
        # plt.ylabel('Miles from Savannah')
        # plt.xlabel('Day')
        # plt.savefig('TrackTest.jpeg', dpi = 1000,  bbox_inches='tight')
        
        # plt.show()
bar.update(15)
print('Writing Unwritten '+str(len(unwrittenFiles))+' Plots')
maxUnwritten = len(unwrittenFiles)
bar = progressbar.ProgressBar(max_value=15)
bar.start()
while True:
    nextUnwrittenFiles = []
    for file  in range(len(unwrittenFiles)):
        try:
            
            unwrittenFiles[file][2].write_html( unwrittenFiles[file][1])
            bar.update(file / maxUnwritten * 15 )
        except IOError as e:
            nextUnwrittenFiles.append(unwrittenFiles[file])
    if len(nextUnwrittenFiles)==0:
        break
    if len(unwrittenFiles) > 1:
        sleep(1)

    unwrittenFiles = nextUnwrittenFiles

bar.update(15)
