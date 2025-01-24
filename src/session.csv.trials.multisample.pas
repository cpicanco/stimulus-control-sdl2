unit session.csv.trials.multisample;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , session.csv.trials.mts;

type

  TNames = array of string;

  { TCSVMultiSample }

  TCSVMultiSample = class(TCSVTrialsMTS)
    private // registered parameters
      FDragDropOrientation: string;
      FReleaseFoodForIntermediateHits : Boolean;
      FAutoAnimateOnstart: Boolean;
      FUseHelpProgression: Boolean;
      FDistance: Integer;
      FDragMode: string;
      FDragMoveFactor: Integer;
      FDragableAnimation: string;
      FGridSize: Integer;
      FRefName : string;
      FName : string;
      FStimuliFolder : string;
      FSampleNames : TNames;
      FComparisonNames : TNames;
      procedure RegisterDynamicParameters(Sender: TObject);
    protected
      procedure AfterLoadingParameters(Sender: TObject); override;
    public
      constructor Create(ASource: string); override;
      property Values[const AKey: string]: string
        read GetValue write SetValue;
  end;

implementation

uses
  sdl.app.trials.dragdrop,
  session.constants.mts,
  session.constants.trials,
  session.constants.trials.dragdrop,
  session.pool;

{ TCSVMultiSample }

procedure TCSVMultiSample.RegisterDynamicParameters(Sender: TObject);
var
  LParametersList : TStringList;
  i : integer = 0;
  LSamples : integer;
  LComparisons : integer;
begin
  LParametersList := Sender as TStringList;
  with MTSKeys do begin
    LSamples := LParametersList.Values[SamplesKey].ToInteger;
    SetLength(FSampleNames, LSamples);
    for i := Low(FSampleNames) to High(FSampleNames) do begin
      RegisterParameter(SampleKey+(i+1).ToString, @FSampleNames[i], '');
    end;

    LComparisons := LParametersList.Values[ComparisonsKey].ToInteger;
    SetLength(FComparisonNames, LComparisons);
    for i := Low(FComparisonNames) to High(FComparisonNames) do begin
      RegisterParameter(ComparisonKey+(i+1).ToString, @FComparisonNames[i], '');
    end;
  end;
end;

procedure TCSVMultiSample.AfterLoadingParameters(Sender: TObject);
begin
  inherited AfterLoadingParameters(Sender);
  if FName.IsEmpty then begin
    FName :=
      TrialID.ToString + '-' +
      Relation + '-' +
      FStimuliFolder + '-' +
      'S' + Samples.ToString + '-' +
      'C' + Comparisons.ToString + '-' +
      'G' + FGridSize.ToString;
  end;

  if FRefName.IsEmpty then begin
    FRefName :=
      Relation + '-' +
      FStimuliFolder + '-' +
      Samples.ToString + '-' +
      Comparisons.ToString;
  end;
end;

constructor TCSVMultiSample.Create(ASource: string);
begin
  inherited Create(ASource);
  OnBeforeLoadingParameters := @RegisterDynamicParameters;
  FKind := TDragDrop.ClassName;
  FDragDropOrientation := '';
  FAutoAnimateOnstart := False;
  FReleaseFoodForIntermediateHits := False;
  FUseHelpProgression := False;
  FDistance := 0;
  FDragMode := '';
  FDragMoveFactor := 0;
  FDragableAnimation := '';
  FGridSize := 0;
  FStimuliFolder := '';
  FName        := '';
  FRefName     := '';

  with TrialKeys, DragDropKeys do begin
    RegisterParameter(ReleaseFoodForIntermediateHitsKey,
      @FReleaseFoodForIntermediateHits, FReleaseFoodForIntermediateHits);
    RegisterParameter(AutoAnimateOnStartKey,
      @FAutoAnimateOnstart, FAutoAnimateOnstart);
    RegisterParameter(DragDropOrientationKey,
      @FDragDropOrientation, FDragDropOrientation);
    RegisterParameter(UseHelpProgressionKey,
      @FUseHelpProgression, FUseHelpProgression);
    RegisterParameter(DistanceKey,
      @FDistance, FDistance);
    RegisterParameter(DragModeKey,
      @FDragMode, FDragMode);
    RegisterParameter(DragMoveFactorKey,
      @FDragMoveFactor, FDragMoveFactor);
    RegisterParameter(DragableAnimationKey,
      @FDragableAnimation, FDragableAnimation);
    RegisterParameter(GridSizeKey,
      @FGridSize, FGridSize);
    RegisterParameter(NameKey,
      @FName, FName);
    RegisterParameter(ReferenceNameKey,
      @FRefName, FRefName);
    RegisterParameter(StimuliFolderKey,
      @FStimuliFolder, FStimuliFolder);
  end;
end;

end.


