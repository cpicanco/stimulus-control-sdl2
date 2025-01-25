{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

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
  , sdl.app.selectable.list
  ;

type

  TState = (startNone, startSamples, startComparisons, startButtons);

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
      //procedure StimulusMouseDown(Sender: TObject; Shift: TCustomShiftState; X, Y: Integer);
      procedure StimulusMouseUp(Sender: TObject; Shift: TCustomShiftState; X, Y: Integer);
      procedure ButtonClick(Sender: TObject);
      procedure NoResponse(Sender: TObject);
      procedure ComparisonResponse(Sender: TObject);
      procedure SampleResponse(Sender: TObject);
    protected
      function CustomName : string; override;
      function MyResult : TTrialResult; override;
      function ToData : string; override;
      function Header : string; override;
    public
      constructor Create; override;
      destructor Destroy; override;
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
  , sdl.app.renderer.validation
  , sdl.app.controls.custom
  , sdl.app.graphics.picture
  //, sdl.app.output
  , sdl.app.audio
  , sdl.app.grids.types
  , sdl.app.grids
  , sdl.app.stimulus.factory
  , sdl.app.stimulus.audio
  , sdl.app.stimulus.speech
  , sdl.app.selectable.contract
  , sdl.app.testmode
  , session.loggers.writerow
  , session.loggers.writerow.timestamp
  , session.constants.trials
  , session.constants.mts
  , session.pool
  , session.strutils
  , session.parameters.global
  ;

{ TMTSStimuli }

function TMTSStimuli.MyResult: TTrialResult;
begin
  Result := FResult;
end;

function TMTSStimuli.ToData: string;
begin
  Result := inherited ToData + #9 + BoolToStr(FHasConsequence, 'True', 'False');
end;

function TMTSStimuli.Header: string;
begin
  Result := inherited Header;
  Result := Result + #9 + 'HasDifferentialReinforcement'
end;

procedure TMTSStimuli.UpdateState(AState: TState);
begin
  FState := AState;
  UpdateNavigator;

  // informs TTrial Owner
  GPaintingInvalidated := True;
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
            FSelectables.Add(LSelectable); // for timestamps only
          end;
        end;
      end;

      startComparisons: begin
        FSelectables.Clear;
        for LStimulus in FSamples do begin
          for LSelectable in LStimulus.Selectables do begin
            FNavigator.SetBaseControl(LSelectable);
            LSelectables.Add(LSelectable);
          end;
        end;

        for LStimulus in FComparisons do begin
          for LSelectable in LStimulus.Selectables do begin
            LSelectables.Add(LSelectable);
            FSelectables.Add(LSelectable);
          end;
        end;
      end;

      startButtons: begin
        FSelectables.Clear;
        for LStimulus in FComparisons do begin
          for LSelectable in LStimulus.Selectables do begin
            LSelectables.Add(LSelectable);
          end;
        end;
        LSelectables.Add(FButton.AsISelectable);
        FSelectables.Add(FButton.AsISelectable);
      end;

    end;

    if LSelectables.Count > 0 then begin
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
    FResponse := LStimulus.AsInterface.ToData;

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
      Finalize;
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
  Finalize;
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

//procedure TMTSStimuli.StimulusMouseDown(Sender: TObject;
//  Shift: TCustomShiftState; X, Y: Integer);
//begin
//
//end;

procedure TMTSStimuli.StimulusMouseUp(Sender: TObject;
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

    ModalityB: begin
      if FMTSModality.Comparisons = ModalityB then begin
        for LIStimulus in FSamples do begin
          LIStimulus.Stop;
        end;
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
  LStimulus: TStimulus;
begin
  case FMTSModality.Samples of
    ModalityA: begin
      if GlobalTrialParameters.SimultaneousMTS then begin
        { do nothing }
      end else begin
        for LStimulus in FSamples do begin
          LStimulus.Hide;
        end;
      end;
    end;

    otherwise begin
      { do nothing }
    end;
  end;

  case FMTSModality.Comparisons of
    ModalityD: begin { Speech }
      SDLAudio.RecorderDevice.Recorder.Stop;
      //TStimulus(Sender).Stop;
    end;

    otherwise begin
      for LStimulus in FComparisons do begin
        LStimulus.Start;
      end;

      UpdateState(startComparisons);
      Timestamp('Comparisons.Start', FSelectables.ToJSON);
    end;
  end;
end;

function TMTSStimuli.CustomName: string;
var
  LSampleModality : string;
  LComparModality : string;
begin
  WriteStr(LSampleModality, FMTSModality.Samples);
  WriteStr(LComparModality, FMTSModality.Comparisons);
  Result := inherited CustomName + '.' +
    LSampleModality + LComparModality.Replace('Modality', '');
end;

procedure TMTSStimuli.NoResponse(Sender: TObject);
begin
  FResult := None;
  Pool.Counters.None;
  //Pool.Counters.Miss;
  Finalize;
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
  TStimulusFactory.Clear;
  FButton.Free;
  FSamples.Free;
  FComparisons.Free;
  inherited Destroy;
end;

function TMTSStimuli.AsINavigable: INavigable;
begin
  Result := Self as INavigable;
end;

procedure TMTSStimuli.DoExpectedResponse;
begin
  if TestMode then begin
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
    Finalize;

  end else begin
    case FState of
      startNone: ;
      startSamples: begin
        FSamples[0].Selectables[0].Select;
        FSamples[0].Selectables[0].Confirm;
      end;
      startComparisons: begin
        FComparisons[0].Selectables[0].Select;
        FComparisons[0].Selectables[0].Confirm;
      end;
      startButtons: begin
        FButton.AsISelectable.Select;
        FButton.AsISelectable.Confirm;
      end;
    end;

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
    //Finalize;
  end;
end;

procedure TMTSStimuli.Load(AParameters: TStringList; AParent: TObject);
var
  LButton      : string;
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
    LFixedComparison: Boolean;
  begin
    LCallbacks.OnMouseExit := @StimulusMouseExit;
    LCallbacks.OnMouseEnter := @StimulusMouseEnter;
    //LCallbacks.OnMouseDown := @StimulusMouseDown;
    LCallbacks.OnMouseUp := @StimulusMouseUp;
    LCallbacks.OnResponse  := @ComparisonResponse;
    LCallbacks.OnNoResponse := @NoResponse;

    if (FMTSModality.Samples = ModalityC) and
       (FMTSModality.Comparisons = ModalityD) then begin
      LFixedComparison := True;
    end else begin
      LFixedComparison := False;
    end;

    Grid.UpdatePositions(Grid.Seed,
      ASamples, AComparisons, AGridOrientation, True, LFixedComparison);

    with Grid.RandomPositions, MTSKeys do begin
      for i := Low(Comparisons) to High(Comparisons) do begin
        LItem := TStimulusFactory.New(Self, FMTSModality.Comparisons, LCallbacks);
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
        LItem := TStimulusFactory.New(Self, FMTSModality.Samples, LCallbacks);
        LItem.IsSample := True;
        LItem.Index := i;
        LItem.Position := Samples[i].Position;

        LItem.Name := SamplesKey+(i+1).ToString;
        LItem.Load(AParameters, AParent, Samples[i].Rect);
        LItem.Sibling := nil;
        Samples[i].Item := LItem as TObject;
        FSamples.Add(LItem);
      end;

      if (FMTSModality.Samples = ModalityC) and
         (FMTSModality.Comparisons = ModalityD) then begin
        FSamples[0].Sibling := FComparisons[0];
      end;

      AppendToTrialHeader(Pool.Session.Trial.Events.Header);
      AppendToTrialHeader(Grid.Header);
      AppendToTrialHeader(Header);
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
        LButton := 'ConfirmButton';
      end;

      ModalityB: begin
        case FMTSModality.Samples of
          ModalityC: begin
            AParameters.Values[HasTextPromptKey] := 'F';
          end;

          otherwise begin
            { do nothing }
          end;
        end;
      end;

      ModalityD : begin
        LButton := 'FinalizeButton';
        LComparisons := 1;
      end;

      otherwise begin
        { do nothing }
      end;
    end;
  end;

  NewGridItems(LSamples, LComparisons, goCustom);

  FButton.LoadFromFile(AsAsset(LButton));
  FButton.Parent := TSDLControl(AParent);
  FButton.OnClick:=@ButtonClick;
  FButton.CustomName := LButton;
  FButton.BoundsRect := Grid.RandomPositions.Samples[0].Rect;
  FButton.ShrinkHeight;
end;

procedure TMTSStimuli.Start;
var
  LStimulus : TStimulus;
begin
  SDLAudio.Clean;

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

