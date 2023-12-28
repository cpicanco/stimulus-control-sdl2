{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.stimuli.mts;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , Generics.Collections
  , sdl.app.graphics.button
  , sdl.app.navigable.contract
  , sdl.app.navigator.contract
  , sdl.app.stimuli.contract
  , sdl.app.stimuli
  , sdl.app.stimulus.contract
  , sdl.app.stimulus
  , sdl.app.trials.types
  , sdl.app.events.abstract
  , sdl.app.audio.contract
  ;

type

  TState = (startNone, startSamples, startComparisons, startButtons);

  TModality = (ModalityNone, ModalityA, ModalityB, ModalityC, ModalityD);

  TMTSModality = record
    Samples     : TModality;
    Comparisons : TModality;
  end;

  TCustomStimulusList = specialize TList<TStimulus>;

  { TMTSStimuli }

  TMTSStimuli = class sealed (TStimuli, INavigable)
    private
      FState : TState;
      FNavigator : ITableNavigator;
      FResult : TTrialResult;
      FHasConsequence : Boolean;
      FSoundCorrect : ISound;
      FSoundWrong   : ISound;
      FMTSModality : TMTSModality;
      FButton : TButton;
      FComparisons : TCustomStimulusList;
      FSamples : TCustomStimulusList;
      procedure UpdateState(AState : TState);
      procedure SetNavigator(ANavigator: ITableNavigator);
      procedure UpdateNavigator;
      procedure DoConsequence(Sender : TObject);
      procedure ConsequenceDone(Sender: TObject);
      procedure ConsequenceStart(Sender: TObject);
      procedure StimulusMouseEnter(Sender: TObject);
      procedure StimulusMouseExit(Sender: TObject);
      procedure StimulusMouseDown(Sender: TObject; Shift: TCustomShiftState; X, Y: Integer);
      procedure StimulusMouseUp(Sender: TObject; Shift: TCustomShiftState; X, Y: Integer);
      procedure ButtonClick(Sender: TObject);
      procedure NoResponse(Sender: TObject);
      procedure ComparisonResponse(Sender: TObject);
      procedure SampleResponse(Sender: TObject);
    public
      constructor Create; override;
      destructor Destroy; override;
      function MyResult : TTrialResult; override;
      function AsIStimuli : IStimuli;
      function AsINavigable : INavigable; override;
      procedure DoExpectedResponse; override;
      procedure Load(AParameters : TStringList;
        AParent : TObject); override;
      procedure Start; override;
      procedure Stop; override;
  end;

implementation

uses
  StrUtils
  , sdl.app.controls.custom
  , sdl.app.graphics.picture
  , sdl.app.output
  , sdl.app.audio
  , sdl.app.grids.types
  , sdl.app.grids
  , sdl.app.stimulus.factory
  , sdl.app.stimulus.audio
  , sdl.app.stimulus.speech
  , sdl.app.selectable.contract
  , sdl.app.selectable.list
  , session.loggers.writerow
  , session.loggers.writerow.timestamp
  , session.constants.trials
  , session.constants.mts
  , session.pool
  , session.strutils
  ;

{ TMTSStimuli }

function TMTSStimuli.MyResult: TTrialResult;
begin
  Result := FResult;
end;

procedure TMTSStimuli.UpdateState(AState: TState);
begin
  FState := AState;
  UpdateNavigator;
end;

procedure TMTSStimuli.SetNavigator(ANavigator: ITableNavigator);
begin
  FNavigator := ANavigator;
end;

procedure TMTSStimuli.UpdateNavigator;
var
  LSelectables : TSelectables;
  LSelectable : ISelectable;
  LStimulus : TStimulus;
begin
  LSelectables := TSelectables.Create;
  try
    case FState of
      startNone: { do nothing };

      startSamples: begin
        for LStimulus in FSamples do begin
          for LSelectable in LStimulus.Selectables do begin
            LSelectables.Add(LSelectable);
          end;
        end;
      end;

      startComparisons: begin
        for LStimulus in FSamples do begin
          for LSelectable in LStimulus.Selectables do begin
            LSelectables.Add(LSelectable);
          end;
        end;

        for LStimulus in FComparisons do begin
          for LSelectable in LStimulus.Selectables do begin
            LSelectables.Add(LSelectable);
          end;
        end;
      end;

      startButtons: begin
        LSelectables.Add(FButton.AsISelectable);
      end;

    end;

    if LSelectables.Count > 0 then begin
      if LSelectables.Count > 1 then begin
        LSelectables.Sort(TSelectables.ByOrigin);
      end;
      FNavigator.UpdateNavigationControls(LSelectables);
    end else begin
      FNavigator.UpdateNavigationControls(nil);
    end;

  finally
    LSelectables.Free;
  end;
end;

procedure TMTSStimuli.DoConsequence(Sender: TObject);
var
  LStimulus : TStimulus;
  LIsHit : Boolean = False;
begin
  if Sender is TStimulus then begin
    LStimulus := Sender as TStimulus;

    if LStimulus is TSpeechStimulus then begin
      LIsHit := LStimulus.IsCorrectResponse;
    end else begin
      LIsHit := LStimulus = FComparisons[0];
    end;

    if LIsHit then begin
      FResult := Hit;
      Pool.Counters.Hit;
    end else begin
      FResult := Miss;
      Pool.Counters.Miss;
    end;

    if FHasConsequence then begin
      if LIsHit then begin
        FSoundCorrect.Play;
      end else begin
        FSoundWrong.Play;
      end;
    end else begin
      if Assigned(OnFinalize) then
        OnFinalize(Self);
    end;
  end;
end;

procedure TMTSStimuli.ConsequenceDone(Sender: TObject);
begin
  if (Sender as ISound) = FSoundCorrect then begin
    Timestamp('Hit.Stop');
  end;

  if (Sender as ISound) = FSoundWrong then begin
    Timestamp('Miss.Stop');
  end;

  FSoundCorrect.SetOnStop(nil);
  FSoundWrong.SetOnStop(nil);
  if Assigned(OnFinalize) then
    OnFinalize(Self);
end;

procedure TMTSStimuli.ConsequenceStart(Sender: TObject);
begin
  if (Sender as ISound) = FSoundCorrect then begin
    Timestamp('Hit.Start');
  end;

  if (Sender as ISound) = FSoundWrong then begin
    Timestamp('Miss.Start');
  end;
end;

procedure TMTSStimuli.StimulusMouseEnter(Sender: TObject);
begin
  //IStimulus(Sender as TStimulus).Start;
end;

procedure TMTSStimuli.StimulusMouseExit(Sender: TObject);
begin
  //IStimulus(Sender as TStimulus).Stop;
end;

procedure TMTSStimuli.StimulusMouseDown(Sender: TObject;
  Shift: TCustomShiftState; X, Y: Integer);
var
  LStimulus: TStimulus;
begin
  if (Sender is TAudioStimulus) then begin
    LStimulus := Sender as TStimulus;
    if not LStimulus.IsSample then begin
      FButton.Hide;
    end;
  end;
end;

procedure TMTSStimuli.StimulusMouseUp(Sender: TObject;
  Shift: TCustomShiftState; X, Y: Integer);
begin

end;

procedure TMTSStimuli.ButtonClick(Sender: TObject);
var
  LStimulus: TStimulus;
begin
  if Sender = FButton then begin
    if FButton.Sender <> nil then begin
      if FButton.Sender is TStimulus then begin
        LStimulus := FButton.Sender as TStimulus;
        DoConsequence(LStimulus);
      end else begin
        raise Exception.Create('TMTSStimuli.ButtonClick');
      end;
    end;
  end;
end;

procedure TMTSStimuli.ComparisonResponse(Sender: TObject);
var
  LStimulus : TStimulus;
  LIStimulus : IStimulus;
begin
  case FMTSModality.Samples of
    ModalityA: begin
      for LIStimulus in FSamples do begin
        LIStimulus.Stop;
      end;
    end;

    otherwise begin
      { do nothing }
    end;
  end;

  if Sender is TStimulus then begin
    //Timestamp('Comparison.Response');
    LStimulus := Sender as TStimulus;

    case FMTSModality.Comparisons of
      ModalityA: begin { TAudioStimulus }
        FButton.Sibling := LStimulus.Rectangule;
        FButton.CentralizeAtTopWith(LStimulus.Rectangule.BoundsRect);
        FButton.Sender := Sender;
        FButton.Show;
        UpdateState(startButtons);
        Exit;
      end;

      ModalityD: begin { TSpeechStimulus }
        FButton.Sibling := LStimulus.Rectangule;
        FButton.CentralizeAtRightWith(LStimulus.Rectangule.BoundsRect);
        FButton.Sender := Sender;
        FButton.Show;
        UpdateState(startButtons);
        Exit;
      end;

      otherwise begin
        { do nothing }
      end;
    end;

    for LIStimulus in FComparisons do begin
      LIStimulus.Stop;
    end;

    DoConsequence(Sender as TStimulus);
  end;
end;

procedure TMTSStimuli.SampleResponse(Sender: TObject);
var
  LStimulus: IStimulus;
begin
  //if FIsDMTS then
  //for LStimulus in FSamples do begin
  //  LStimulus.Stop;
  //end;

  for LStimulus in FComparisons do begin
    LStimulus.Start;
  end;
  Timestamp('Comparison.Start');
  UpdateState(startComparisons);
end;

procedure TMTSStimuli.NoResponse(Sender: TObject);
begin
  FResult := None;
  Pool.Counters.None;
  //Pool.Counters.Miss;
  if Assigned(OnFinalize) then
    OnFinalize(Self);
end;


constructor TMTSStimuli.Create;
begin
  inherited Create;
  FButton := TButton.Create;
  FSamples := TCustomStimulusList.Create;
  FComparisons := TCustomStimulusList.Create;
  FMTSModality.Samples := ModalityNone;
  FMTSModality.Comparisons := ModalityNone;

  TStimulusFactory.Clear;
end;

destructor TMTSStimuli.Destroy;
begin
  UpdateState(startNone);
  TStimulusFactory.Clear;
  FButton.Free;
  FSamples.Free;
  FComparisons.Free;
  inherited Destroy;
end;

function TMTSStimuli.AsIStimuli: IStimuli;
begin
  Result := Self as IStimuli;
end;

function TMTSStimuli.AsINavigable: INavigable;
begin
  Result := Self as INavigable;
end;

procedure TMTSStimuli.DoExpectedResponse;
begin
  // for real time simulations
  //case High(FComparisons) of
  //  0 : FComparisons[0].DoResponse(False);
  //  else begin
  //    if Random < (0.9/1.0) then begin
  //      FComparisons[0].DoResponse(False);
  //    end else begin
  //      FComparisons[RandomRange(1, Length(FComparisons))].DoResponse(False);
  //    end;
  //  end;
  //end;
  //if Assigned(OnFinalize) then
  //  OnFinalize(Self);

  // for short time simulations
  case FComparisons.Count-1 of
    -MaxInt..-1: { do nothing };
    0 : Pool.Counters.Hit;
    else begin
      if Random < (0.9/1.0) then begin
        Pool.Counters.Hit;
      end else begin
        Pool.Counters.Miss;
      end;
    end;
  end;
  if Assigned(OnFinalize) then
    OnFinalize(Self);
end;

procedure TMTSStimuli.Load(AParameters: TStringList; AParent: TObject);
var
  LRelation    : string;
  SampleLetter : string;
  ComparLetter : string;
  LSamples      : integer;
  LComparisons  : integer;
  //LCycle : string;

  procedure NewGridItems(ASamples, AComparisons: integer;
    AGridOrientation: TGridOrientation);
  var
    LItem : TStimulus;
    i : integer;
    LCallbacks : TCallbacks;
  begin
    LCallbacks.OnMouseExit := @StimulusMouseExit;
    LCallbacks.OnMouseEnter := @StimulusMouseEnter;
    LCallbacks.OnMouseDown := @StimulusMouseDown;
    LCallbacks.OnMouseUp := @StimulusMouseUp;
    LCallbacks.OnResponse  := @ComparisonResponse;
    LCallbacks.OnNoResponse := @NoResponse;

    if not Assigned(Grid) then begin
      Grid := TGrid.Create(3);
    end;
    Grid.FixedSample:=True;

    if AComparisons = 1 then begin
      Grid.FixedComparison:=True;
    end else begin
      Grid.FixedComparison:=False;
    end;

    Grid.UpdatePositions(ASamples, AComparisons, AGridOrientation);
    with Grid.RandomPositions, MTSKeys do begin
      for i := Low(Comparisons) to High(Comparisons) do begin
        LItem := TStimulusFactory.New(Self, ComparLetter, LCallbacks);
        LItem.IsSample := False;
        LItem.Index := i;
        LItem.Position := Comparisons[i].Position;

        LItem.Name:=ComparisonsKey+(i+1).ToString;
        LItem.Load(AParameters, AParent, Comparisons[i].Rect);

        Comparisons[i].Item := LItem as TObject;
        FComparisons.Add(LItem);
      end;

      LCallbacks.OnResponse := @SampleResponse;
      for i := Low(Samples) to High(Samples) do begin
        LItem := TStimulusFactory.New(Self, SampleLetter, LCallbacks);
        LItem.IsSample := True;
        LItem.Index := i;
        LItem.Position := Samples[i].Position;

        LItem.Name := SamplesKey+(i+1).ToString;
        LItem.Load(AParameters, AParent, Samples[i].Rect);

        Samples[i].Item := LItem as TObject;
        FSamples.Add(LItem);
      end;
      AppendToTrialHeader(Pool.Session.Trial.Events.Header);
      AppendToTrialHeader(Grid.Header);
    end;
  end;

begin
  if not Assigned(AParent) then
    raise Exception.Create('You must assign a parent.');
  FSoundCorrect := SDLAudio.SoundFromName('acerto');
  FSoundWrong   := SDLAudio.SoundFromName('erro');
  FSoundCorrect.SetOnStart(@ConsequenceStart);
  FSoundWrong.SetOnStart(@ConsequenceStart);
  FSoundCorrect.SetOnStop(@ConsequenceDone);
  FSoundWrong.SetOnStop(@ConsequenceDone);

  with TrialKeys, MTSKeys do begin
    FHasConsequence := AParameters.Values[HasConsequenceKey].ToBoolean;
    LRelation := AParameters.Values[RelationKey];
    SampleLetter := ExtractDelimited(1,LRelation,['-']);
    ComparLetter := ExtractDelimited(2,LRelation,['-']);
    LSamples := AParameters.Values[SamplesKey].ToInteger;
    LComparisons := AParameters.Values[ComparisonsKey].ToInteger;

    case SampleLetter of
      'A' : FMTSModality.Samples := ModalityA;
      'B' : FMTSModality.Samples := ModalityB;
      'C' : FMTSModality.Samples := ModalityC;
      'D' : FMTSModality.Samples := ModalityD;
      else
        raise Exception.Create('Unknown Samples modality: ' + SampleLetter);
    end;
    case ComparLetter of
      'A' : FMTSModality.Comparisons := ModalityA;
      'B' : FMTSModality.Comparisons := ModalityB;
      'C' : FMTSModality.Comparisons := ModalityC;
      'D' : FMTSModality.Comparisons := ModalityD;
      else
        raise Exception.Create('Unknown Comparisons modality: ' + SampleLetter);
    end;

    case FMTSModality.Comparisons of
      ModalityA : begin
        FButton.LoadFromFile(AsAsset('ConfirmButton'));
        FButton.Parent := TSDLControl(AParent);
        FButton.OnClick:=@ButtonClick;
      end;
      ModalityD : begin
        FButton.LoadFromFile(AsAsset('FinalizeButton'));
        FButton.Parent := TSDLControl(AParent);
        FButton.OnClick:=@ButtonClick;

        LComparisons := 1;
      end;
      else { do nothing }
    end;
  end;

  NewGridItems(LSamples, LComparisons, goCustom);
end;

procedure TMTSStimuli.Start;
var
  LStimulus : TStimulus;
begin
  for LStimulus in FSamples do
    LStimulus.Start;
  UpdateState(startSamples);
end;

procedure TMTSStimuli.Stop;
var
  LStimulus : TStimulus;
begin
  for LStimulus in FComparisons do
    LStimulus.Stop;

  for LStimulus in FSamples do
    LStimulus.Stop;
end;

end.

