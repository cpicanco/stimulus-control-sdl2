{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço, Universidade Federal do Pará.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.helpseries.dragdrop;

{$mode ObjFPC}{$H+}

{$modeswitch TypeHelpers}
{$modeswitch AdvancedRecords}

interface

uses Classes, SysUtils
   //, sdl.app.graphics.picture.dragdrop
   , generics.aggregator.contract
   , generics.aggregator
   ;

type

  TEquivalenceRelation =
    (eqrAA, eqrBB, eqrCC, eqrAB, eqrBA, eqrAC, eqrCA, eqrBC, eqrCB);
  TEquivalenceRelationRange = eqrAA..eqrCB;

  TSampleValue = (sampOne, sampTwo, sampThree);
  TSampleRange = sampOne..sampThree;

  TComparValue = (compOne, compTwo, compThree);
  TComparRange = compOne..compThree;

  TFactor = (facVeryEasy, facEasy, facNormal, facHard, facVeryHard);
  TFactorRange = facVeryEasy..facVeryHard;

  TDragDropOrientation = (goTopToBottom, goBottomToTop, goLeftToRight,
                     goRightToLeft, goRandom);
  TDragDropOrientationRange = goTopToBottom..goRandom;

  TDistanceValue = (distZero, distTen, distTwenty, distThirty, distForty, distFifty);
  TDistanceRange = distZero..distFifty;

  TDragDropData = record
    Relation : TEquivalenceRelation;
    Samples: TSampleValue;
    Comparisons: TComparValue;
    //HelpType: TDragMouseMoveMode;
    Factor: TFactor;
    Orientation : TDragDropOrientation;
    Distance : TDistanceValue;
    class operator = (A, B: TDragDropData): Boolean;
  end;

type
  THelpSerie = (hsFromFile,
    hsDefault, hsPreTraining, hsIdentityTraining, hsArbitraryTraining);

  TTDragDropHelp = specialize TAggregator<TDragDropData>;

  { TDragDropHelpSerie }

  TDragDropHelpSerie = class sealed (TTDragDropHelp)
    private
      FHelpSerieKind : THelpSerie;
      procedure LoadDefaultHelpSerie;
      procedure LoadPreTrainingHelpSerie;
      procedure LoadIdentityTrainingHelpSerie;
      procedure LoadArbitraryTrainingHelpSerie;
    protected
      procedure LoadFromMock; override;
      procedure LoadFromFile(AFilename : string); override;
      procedure SaveToFile(AFilename : string = '');
    public
      constructor Create(AHelpSerie : THelpSerie); overload;
      constructor Create(AFilename  : string); overload;
      procedure AssignCurrent(AParameters : TStringList); override;
      procedure AssignParameters(AParameters : TStringList); override;
  end;

type

  { TEquivalenceRelationHelper }

  TEquivalenceRelationHelper = type helper for TEquivalenceRelation
    function ToString : string;
  end;

  { TSampleValueHelper }

  TSampleValueHelper = type helper for TSampleValue
    function ToString : string;
    function ToInteger: integer;
  end;

  { TComparValueHelper }

  TComparValueHelper = type helper for TComparValue
    function ToString : string;
    function ToInteger: integer;
  end;

  { TFactorHelper }

  TFactorHelper = type helper for TFactor
    function ToString : string;
    function ToInteger: integer;
  end;

  { TDistanceValueHelper }

  TDistanceValueHelper = type helper for TDistanceValue
    function ToString : string;
    function ToInteger: integer;
  end;

  { TDragDropOrientationHelper }

  TDragDropOrientationHelper = type helper for TDragDropOrientation
    function ToString : string;
    //function ToInteger : integer;
  end;

  { THelpSeriesStringHelper }

  THelpSeriesStringHelper = type helper(TStringHelper) for string
    function ToEquivalenceRelation : TEquivalenceRelation;
    function ToSampleValue : TSampleValue;
    function ToComparValue : TComparValue;
    function ToFactor : TFactor;
    function ToDragDropOrientation : TDragDropOrientation;
    function ToDistanceValue : TDistanceValue;
  end;

    { THelpSeriesIntegerHelper }

    THelpSeriesIntegerHelper = type helper(TIntegerHelper) for integer
      function ToFactor : TFactor;
    end;

var
  DefaultDragDropData : TDragDropData;
  DragDropHelpSerie : TDragDropHelpSerie;
  IDragDropHelpSerie : specialize IAggregator<TDragDropData>;

const
  DefaultComplexityFilename : string = 'ComplexityGradient.ini';

implementation

uses
  IniFiles
  , session.pool
  , session.constants.dragdrop
  ;

class operator TDragDropData.=(A, B : TDragDropData) : Boolean;
begin
  Result :=
    (A.Relation = B.Relation) and
    (A.Samples = B.Samples) and
    (A.Comparisons = B.Comparisons) and
    //(A.HelpType = B.HelpType) and
    (A.Factor = B.Factor);
end;

{ TDragDropHelpSerie }

procedure TDragDropHelpSerie.LoadDefaultHelpSerie;
var
  LDragDropData : TDragDropData;
begin
  with LDragDropData do begin
    Relation := eqrAA;
    //HelpType := dragChannel;
    Samples := sampOne;
    Comparisons := compOne;
    Factor := facVeryEasy;
  end;
  FList.Add(LDragDropData);

  LDragDropData.Factor := facEasy;
  FList.Add(LDragDropData);

  LDragDropData.Factor := facNormal;
  FList.Add(LDragDropData);

  LDragDropData.Factor := facHard;
  FList.Add(LDragDropData);

  LDragDropData.Factor := facVeryHard;
  FList.Add(LDragDropData);
end;

procedure TDragDropHelpSerie.LoadPreTrainingHelpSerie;
begin

end;

procedure TDragDropHelpSerie.LoadIdentityTrainingHelpSerie;
begin

end;

procedure TDragDropHelpSerie.LoadArbitraryTrainingHelpSerie;
begin

end;

constructor TDragDropHelpSerie.Create(AHelpSerie: THelpSerie);
begin
  inherited Create;
  FHelpSerieKind := AHelpSerie;
  LoadFromMock;
  //SaveToFile;
end;

constructor TDragDropHelpSerie.Create(AFilename: string);
begin
  inherited Create;
  FHelpSerieKind := hsFromFile;
  LoadFromFile(AFilename);
end;

procedure TDragDropHelpSerie.AssignCurrent(AParameters: TStringList);
var
  Data : TDragDropData;
  FCurrentIndex : integer;
begin
  with AParameters do begin
    with DragDropKeys do begin
      //Data.HelpType := Values[SamplesDragMode].ToDragMouseMoveMode;
      Data.Relation := Values[Relation].ToEquivalenceRelation;
      Data.Samples := Values[Samples].ToSampleValue;
      Data.Comparisons := Values[Comparisons].ToComparValue;
      Data.Factor := Values[DragMoveFactor].ToInteger.ToFactor;
      Data.Orientation := Values[DragDropOrientation].ToDragDropOrientation;
      Data.Distance := Values[Distance].ToDistanceValue;
    end;
  end;

  FCurrentIndex := Iterator.IndexOf(Data);
  if FCurrentIndex = -1 then begin
    List.Add(Data);
  end else begin
    Iterator.SetCurrent(FCurrentIndex);
  end;
end;

procedure TDragDropHelpSerie.AssignParameters(AParameters: TStringList);
var
  Data : TDragDropData;
begin
  Data := Iterator.GetCurrent;
  with AParameters do begin
    with DragDropKeys do begin
      //Values[SamplesDragMode] := Data.HelpType.ToString;
      Values[Relation] := Data.Relation.ToString;
      Values[Samples] := Data.Samples.ToString;
      Values[Comparisons] := Data.Comparisons.ToString;
      Values[DragMoveFactor] := Data.Factor.ToInteger.ToString;
      Values[DragDropOrientation] := Data.Orientation.ToString;
      Values[Distance] := Data.Distance.ToString;
    end;
  end;
end;

procedure TDragDropHelpSerie.LoadFromMock;
begin
  case FHelpSerieKind of
    hsFromFile: { do nothing };
    hsDefault : LoadDefaultHelpSerie;
    hsPreTraining : LoadPreTrainingHelpSerie;
    hsIdentityTraining : LoadIdentityTrainingHelpSerie;
    hsArbitraryTraining: LoadArbitraryTrainingHelpSerie;
  end;
end;

procedure TDragDropHelpSerie.LoadFromFile(AFilename: string);
var
  LIniFile : TIniFile;
  i: integer = 0;

  function GetSection : string;
  begin
    Result := 'Level' + #32 + (i+1).ToString;
  end;

  function DragDropDataFromSection(ASection : string) : TDragDropData;
  begin
    with DragDropKeys do begin
      //Result.HelpType :=
      //  LIniFile.ReadString(
      //    ASection, SamplesDragMode, 'Default').ToDragMouseMoveMode;
      Result.Relation :=
        LIniFile.ReadString(
          ASection, Relation, 'Default').ToEquivalenceRelation;
      Result.Samples :=
        LIniFile.ReadString(
          ASection, Samples, 'Default').ToSampleValue;
      Result.Comparisons :=
        LIniFile.ReadString(
          ASection, Comparisons, 'Default').ToComparValue;
      Result.Factor :=
        LIniFile.ReadString(
          ASection, DragMoveFactor, 'Default').ToInteger.ToFactor;
      Result.Orientation :=
        LIniFile.ReadString(
          ASection, DragDropOrientation, 'Default').ToDragDropOrientation;
      Result.Distance :=
        LIniFile.ReadString(
          ASection, Distance, 'Default').ToDistanceValue;
    end;
    Inc(i);
  end;

begin
  FList.Clear;
  LIniFile := TIniFile.Create(AFileName);
  try
    while LIniFile.SectionExists(GetSection) do
      FList.Add(DragDropDataFromSection(GetSection));
  finally
    LIniFile.Free;
  end;
end;

procedure TDragDropHelpSerie.SaveToFile(AFilename: string);
var
  Data : TDragDropData;
  IniFile : TIniFile;
  Filename : string;
  i: integer = 0;

  function GetSection : string;
  begin
    Result := 'Level' + #32 + (i+1).ToString;
  end;
begin
  if AFilename = '' then begin
    Filename := Pool.BaseFilePath + DefaultComplexityFilename;
  end else begin
    Filename := AFilename;
  end;

  IniFile := TIniFile.Create(Filename);
  for Data in List do begin
    with DragDropKeys do begin
      with IniFile do begin
        //WriteString(GetSection,
        //  SamplesDragMode, Data.HelpType.ToString);
        WriteString(GetSection,
          Relation, Data.Relation.ToString);
        WriteString(GetSection,
          Samples, Data.Samples.ToString);
        WriteString(GetSection,
          Comparisons, Data.Comparisons.ToString);
        WriteString(GetSection,
          DragMoveFactor, Data.Factor.ToInteger.ToString);
        WriteString(GetSection,
          DragDropOrientation, Data.Orientation.ToString);
        WriteString(GetSection,
          Distance, Data.Distance.ToString);
      end;
    end;
    Inc(i);
  end;
  IniFile.Free;
end;

{ TEquivalenceRelationHelper }

function TEquivalenceRelationHelper.ToString: string;
var
  Relations : array [TEquivalenceRelationRange] of string =
    ('A-A', 'B-B', 'C-C', 'A-B', 'B-A', 'A-C', 'C-A', 'B-C', 'C-B');
begin
  Result := Relations[Self];
end;

{ TSampleValueHelper }

function TSampleValueHelper.ToString: string;
begin
  case Self of
    sampOne   : Result := '1';
    sampTwo   : Result := '2';
    sampThree : Result := '3';
  end;
end;

function TSampleValueHelper.ToInteger: integer;
var
  SampleValues : array [TSampleRange] of integer = (1, 2, 3);
begin
  Result := SampleValues[Self];
end;

{ TComparValueHelper }

function TComparValueHelper.ToString: string;
begin
  case Self of
    compOne   : Result := '1';
    compTwo   : Result := '2';
    compThree : Result := '3';
  end;
end;

function TComparValueHelper.ToInteger: integer;
var
  ComparValues : array [TComparRange] of integer = (1, 2, 3);
begin
  Result := ComparValues[Self];
end;

{ TFactorHelper }

function TFactorHelper.ToString: string;
begin
  WriteStr(Result, Self);
  Result := Result.Replace('fac', '');
end;

var
  Factors   : array [TFactorRange] of integer = (16, 8, 5, 4, 2);

function TFactorHelper.ToInteger: integer;
begin
  Result := Factors[Self];
end;

{ TDistanceValueHelper }

function TDistanceValueHelper.ToString: string;
begin
  case Self of
    distZero   : Result := '0';
    distTen    : Result := '10';
    distTwenty : Result := '20';
    distThirty : Result := '30';
    distForty  : Result := '40';
    distFifty  : Result := '50';
  end;
end;

function TDistanceValueHelper.ToInteger: integer;
var
  DistanceValues : array [TDistanceRange] of integer = (0, 10, 20, 30, 40, 50);
begin
  Result := DistanceValues[Self];
end;

{ TOrientationHelper }

function TDragDropOrientationHelper.ToString : string;
begin
  WriteStr(Result, Self);
  Result := Result.Replace('go', '');
end;

{ THelpSeriesStringHelper }

function THelpSeriesStringHelper.ToEquivalenceRelation: TEquivalenceRelation;
begin
  case UpperCase(Self) of
    'A-A', 'A->A', 'AA' : Result := eqrAA;
    'B-B', 'B->B', 'BB' : Result := eqrBB;
    'C-C', 'C->C', 'CC' : Result := eqrCC;
    'A-B', 'A->B', 'AB' : Result := eqrAB;
    'B-A', 'B->A', 'BA' : Result := eqrBA;
    'A-C', 'A->C', 'AC' : Result := eqrAC;
    'C-A', 'C->A', 'CA' : Result := eqrCA;
    'B-C', 'B->C', 'BC' : Result := eqrBC;
    'C-B', 'C->B', 'CB' : Result := eqrCB;
    'DEFAULT' : Result := DefaultDragDropData.Relation;
    else
      raise Exception.Create(
        'THelpSeriesStringHelper.ToEquivalenceRelation: ' + Self);
  end;
end;

function THelpSeriesStringHelper.ToSampleValue: TSampleValue;
begin
  case UpperCase(Self) of
    '1' : Result := sampOne;
    '2' : Result := sampTwo;
    '3' : Result := sampThree;
    'DEFAULT' : Result := DefaultDragDropData.Samples;
    else
      raise Exception.Create(
        'THelpSeriesStringHelper.ToSampleValue: ' + Self);
  end;
end;

function THelpSeriesStringHelper.ToComparValue: TComparValue;
begin
  case UpperCase(Self) of
    '1' : Result := compOne;
    '2' : Result := compTwo;
    '3' : Result := compThree;
    'DEFAULT' : Result := DefaultDragDropData.Comparisons;
    else
      raise Exception.Create(
        'THelpSeriesStringHelper.ToComparValue: ' + Self);
  end;
end;

function THelpSeriesStringHelper.ToFactor: TFactor;
begin
  case UpperCase(Self) of
    'VERYEASY' : Result := facVeryEasy;
    'EASY' : Result := facEasy;
    'NORMAL' : Result := facNormal;
    'HARD' : Result := facHard;
    'VERYHARD' : Result := facVeryHard;
    'DEFAULT' : Result := DefaultDragDropData.Factor;
    else
      raise Exception.Create(
        'THelpSeriesStringHelper.ToFactor: ' + Self);
  end;
end;

function THelpSeriesStringHelper.ToDragDropOrientation: TDragDropOrientation;
begin
  case UpperCase(Self) of
    'TOPTOBOTTOM' : Result := goTopToBottom;
    'BOTTOMTOTOP' : Result := goBottomToTop;
    'LEFTTORIGHT' : Result := goLeftToRight;
    'RIGHTTOLEFT' : Result := goRightToLeft;
    'RANDOM' : Result := goRandom;
    'DEFAULT' : Result := DefaultDragDropData.Orientation;
    else
      raise Exception.Create(
        'THelpSeriesStringHelper.ToOrientation: ' + Self);
  end;
end;

function THelpSeriesStringHelper.ToDistanceValue: TDistanceValue;
begin
  case UpperCase(Self) of
    '0' : Result := distZero;
    '10' : Result := distTen;
    '20' : Result := distTwenty;
    '30' : Result := distThirty;
    '40' : Result := distForty;
    '50' : Result := distFifty;
    'DEFAULT' : Result := DefaultDragDropData.Distance;
    else
      raise Exception.Create(
        'THelpSeriesStringHelper.ToDistanceValue: ' + Self);
  end;
end;

{ THelpSeriesIntegerHelper }

function THelpSeriesIntegerHelper.ToFactor: TFactor;
begin
  for Result in TFactorRange do
    if Factors[Result] = Self then
      Break;
end;

end.


