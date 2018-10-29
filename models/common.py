"""model common helper"""

# Authorship
__author__ = "Ji Li"
__email__ = "jili5@microsoft.com"

import secret

# print(secret.MSRA_URL TAS_URL file_icon_img_mapping)

# load icon description-img name mapping
def load_dict(f):
	d = {}
	with open(f, 'r') as fp:
		for line in fp:
			line = line.split()
			d[' '.join(line[1:])] = line[0][:-4]
	d["Right Pointing Backhand Index "] = d["Right Pointing Backhand Index"]
	return d
mp_icondescription2filename = load_dict(secret.file_icon_img_mapping)

ML_query_template = {"PresentationInputMetaData":{"SlideInputMetaData":[{"SlideId":0,"RunNlxAnalysis":False,"RunTextBoxAnalysis":False,"RunIconAnalysis":True,"RunPercentageAnalysis":False,"RunAgendaAnalysis":False,"ShapeInputMetaData":[{"ShapeId":7,"RunCeousAnalysis":False,"RunIconChunkingAnalysis":False,"ParagraphInputMetaData":[],"RunSmartArtAnalysis":False,"Hints":{}}],"Hints":{}}],"Hints":{"SlideSize_X":"12192000","SlideSize_Y":"6858000"}},"TextInputData":{"Title":"Clock is important","LayoutId":0,"Pictures":[],"SmartArts":[],"TextBoxes":[{"Texts":[{"Text":"What is in the context","Bullet":"DefaultBullet","Level":0,"IsBold":False,"IsItalics":False,"IsUnderline":False,"IsStrikeThrough":False,"FontName":None,"FontSize":2800.0,"TextRuns":[{"StartIndex":0,"Count":22,"Language":"en-US"}]}],"TextBullet":None,"TextDirection":None,"TextSpacingPercent":0,"TextSpacingPoint":0,"IsPh":True,"IsTextBox":False,"PhType":None,"ShapeId":7,"Left":838200,"Top":1825625,"Right":11353800,"Bottom":6176963,"Width":10515600,"Height":4351338}],"Charts":[]},"Hints":{"AllowChartSuggestions":"true","IsUnifiedPrivacyAccepted":"true","SlidesCount":"6","AudienceGroup":"Production","DocumentId":"{26A282B8-3118-4A3B-8F0A-BF910E318584}","DeferredFlights":"of2cpdsdfph3dg20","AudienceChannel":"CC","IsLegacyPrivacyAccepted":"true","SlideIndex":"4","ApplyLogicFlags":"3","FeatureGates":"F","PaneOpen":"false","LocalTime":"2018-07-10T16:54:17.740-07:00","DocumentLocation":"16","AllowMulParasWithAnim":"true","DocCreateTime":"2018-06-18T23:29:43","IsAuthor":"true","TenantId":"72f988bf-86f1-41af-91ab-2d7cd011db47","ProtocolVer":"2","IdentityProvider":"ADAL","CID":"479851ff6b8563c06922bcadf6ba610807895631b757288d6b698186bd3435a4df61f99ad758b52694f66f420f64e76de2c4ce50fbaab7b315f9336bb9d2d712","IconChunking":"7","FlightID":"5E48086F-661A-4F9C-B3AE-65268FDDC5D2","CommandsClicked":"v=1;26891,268437248,3882823;26891,268437248,3883424;26891,268437248,3896384;26891,268437248,3896994;26891,268437248,3995070;26891,268437248,3995471;26891,268437248,6705922;26891,268437248,6707156;26891,268437248,6786437;26891,268437248,6787006;26891,268437248,7187609;26891,268437248,7188105;26891,268437248,7200219;26891,268437248,7200739;26891,268437248,7314277;26891,268437248,7314705;26891,268437248,7351043;26891,268437248,7351477;26891,268437248,7481675;26891,268437248,7481972;26891,268437248,7505095;26891,268437248,7505601;11525,268442624,7508702;11525,268442624,7509514;11525,268442624,7509514;26891,268437248,7621987;26891,268437248,7622343;26891,268437248,7704928;26891,268437248,7708708;26891,268437248,7722794;26891,268437248,7773133;26891,268437248,7773650;26891,268437248,7819098;26891,268437248,7830309;11527,268444672,3813396;11527,268444672,3813397;11954,268444672,3816210;11527,268444672,3824890;11527,268444672,3824891;11954,268444672,3825387;11527,268444672,3835681;11527,268444672,3835682;11954,268444672,3836201;11527,268444672,3838168;11527,268444672,3838169;11954,268444672,3838593;26891,268437248,3846898;26891,268437248,3848566;26891,268437248,3850492;13955,268442112,3881638","Locale":"en-US","SlideId":"260","SessionID":"{6EF3B2F6-B78E-45C6-A161-22BAE49088F2}","MachineID":"{7901D5B8-95F7-4EC3-B917-8AD4E4C524E0}","Trigger":"ToolbarButton","Endpoint":"powerpnt.exe","DayOfWeek":"Tuesday","TimeOfDay":"1434","LocalDayOfWeek":"Tuesday","LocalTimeOfDay":"1014","EndpointName":"powerpnt.exe","OriginalSlideCID":"2362652297","IconChunkingInterpretedTextEnabled":"False"}}

MSRA_query_template = {'max_keywords_length':15, 
'keywords': ""}

