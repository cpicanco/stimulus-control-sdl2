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
  , fgl
  , sdl.app.graphics.button
  , sdl.app.stimuli
  , sdl.app.stimulus.contract
  , sdl.app.stimuli.contract
  , sdl.app.events.abstract
  , sdl.app.audio.contract
  ;

type

  TModality = (ModalityNone, ModalityA, ModalityB, ModalityC, ModalityD);

  TMTSModality = record
    Samples     : TModality;
    Comparisons : TModality;
  end;

  TIStimulusList = specialize TFPGList<IStimulus>;

  { TMTSStimuli }

  TMTSStimuli = class sealed (TStimuli, IStimuli)
    private
      FMTSModality : TMTSModality;
      FHasConsequence : Boolean;
      FButton : TButton;
      FSoundCorrect : ISound;
      FSoundWrong   : ISound;
      FComparisons : TIStimulusList;
      FSamples : TIStimulusList;
      procedure DoConsequence(Sender : TObject);
      procedure ConsequenceDone(Sender: TObject);
      procedure StimulusMouseEnter(Sender: TObject);
      procedure StimulusMouseExit(Sender: TObject);
      procedure StimulusMouseDown(Sender: TObject; Shift: TCustomShiftState; X, Y: Integer);
      procedure StimulusMouseUp(Sender: TObject; Shift: TCustomShiftState; X, Y: Integer);
      procedure ButtonClick(Sender: TObject);
      procedure ComparisonResponse(Sender: TObject);
      procedure SampleResponse(Sender: TObject);
    public
      constructor Create(AOwner : TComponent); override;
      destructor Destroy; override;
      function AsInterface : IStimuli;
      procedure DoExpectedResponse; override;
      procedure Load(AParameters : TStringList;
        AParent : TObject); override;
      procedure Start; override;
      procedure Stop; override;
  end;

implementation

uses
  StrUtils
  , sdl.app.renderer.custom
  , sdl.app.graphics.picture
  , sdl.app.output
  , sdl.app.audio
  , sdl.app.grids.types
  , sdl.app.grids
  , sdl.app.stimulus.factory
  , sdl.app.stimulus
  , sdl.app.stimulus.audio
  , sdl.app.stimulus.speech
  , session.constants.trials
  , session.constants.mts
  , session.pool
  ;

{ TMTSStimuli }

procedure TMTSStimuli.DoConsequence(Sender: TObject);
var
  LStimulus : TStimulus;
  LIsHit : Boolean = False;
begin
  if FHasConsequence then begin

    if Sender is TStimulus then begin
      LStimulus := Sender as TStimulus;

      if LStimulus is TSpeechStimulus then begin
        LIsHit := LStimulus.IsCorrectResponse;
      end else begin
        LIsHit := LStimulus = FComparisons[0];
      end;

      if LIsHit then begin
        FSoundCorrect.Play;
        Pool.Counters.Hit;
      end else begin
        FSoundWrong.Play;
        Pool.Counters.Miss;
      end;
    end;

  end else begin

    if Assigned(OnFinalize) then
      OnFinalize(Self);

  end;
end;

procedure TMTSStimuli.ConsequenceDone(Sender: TObject);
begin
  FSoundCorrect.SetOnStop(nil);
  FSoundWrong.SetOnStop(nil);
  if Assigned(OnFinalize) then
    OnFinalize(Self);
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
    if FButton.Sibling <> nil then begin
      FButton.Hide;
      LStimulus := FButton.Sibling.Owner as TStimulus;
      DoConsequence(LStimulus);
    end;
  end;
end;

procedure TMTSStimuli.ComparisonResponse(Sender: TObject);
var
  LStimulus : TStimulus;
  LIStimulus : IStimulus;
begin
  if Sender is TStimulus then begin
    LStimulus := Sender as TStimulus;
    if LStimulus is TSpeechStimulus then begin
      FButton.Sibling := TSpeechStimulus(LStimulus).Rectangule;
      FButton.Show;
      Exit;
    end;

    if LStimulus is TAudioStimulus then begin
      FButton.Sibling := TAudioStimulus(LStimulus).Rectangule;
      FButton.Show;
      Exit;
    end;

    for LIStimulus in FComparisons do begin
      LIStimulus.Stop;
    end;

    DoConsequence(LStimulus);
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
end;


constructor TMTSStimuli.Create(AOwner: TComponent);
begin
  inherited Create(AOwner);
  FSamples := TIStimulusList.Create;
  FComparisons := TIStimulusList.Create;
  FMTSModality.Samples := ModalityNone;
  FMTSModality.Comparisons := ModalityNone;
  FButton:= nil;
end;

destructor TMTSStimuli.Destroy;
begin
  FSamples.Free;
  FComparisons.Free;
  if Assigned(FButton) then
    FButton.Free;

  if FMTSModality.Comparisons = ModalityD then begin
    if Assigned(SDLAudio) then begin
      SDLAudio.RecorderDevice.Close;
    end;
  end;
  inherited Destroy;
end;

function TMTSStimuli.AsInterface: IStimuli;
begin
  Result := Self as IStimuli
end;

procedure TMTSStimuli.DoExpectedResponse;
begin
  //FComparisons[0].DoResponse;
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
    LParameters : TStringList;
    LCallbacks : TCallbacks;
  begin
    LCallbacks.OnMouseExit := @StimulusMouseExit;
    LCallbacks.OnMouseEnter := @StimulusMouseEnter;
    LCallbacks.OnMouseDown := @StimulusMouseDown;
    LCallbacks.OnMouseUp := @StimulusMouseUp;
    LCallbacks.OnResponse  := @ComparisonResponse;
    LParameters := TStringList.Create;
    try
      if not Assigned(Grid) then
        Grid := TGrid.Create(3);
      Grid.FixedSample:=True;

      if AComparisons = 1 then begin
        Grid.FixedComparison:=True;
      end else begin
        Grid.FixedComparison:=False;
      end;

      Grid.UpdatePositions(ASamples, AComparisons, AGridOrientation);
      with Grid.RandomPositions do begin
        for i := Low(Comparisons) to High(Comparisons) do begin
          LItem := TStimulusFactory.New(Self, ComparLetter, LCallbacks);
          LItem.IsSample := False;
          LItem.Index := i;

          LParameters.Clear;
          LItem.Name:=MTSKeys.Comparisons+(i+1).ToString;
          LItem.Load(AParameters, AParent, Comparisons[i].Rect);

          Comparisons[i].Item := LItem as TObject;
          FComparisons.Add(LItem);
        end;

        LCallbacks.OnResponse := @SampleResponse;
        for i := low(Samples) to high(Samples) do begin
          LItem := TStimulusFactory.New(Self, SampleLetter, LCallbacks);
          LItem.IsSample := True;
          LItem.Index := i;

          LItem.Name := MTSKeys.Samples+(i+1).ToString;
          LItem.Load(AParameters, AParent, Samples[i].Rect);

          Samples[i].Item := LItem as TObject;
          FSamples.Add(LItem);
        end;
      end;
    finally
      LParameters.Free;
    end;
  end;

begin
  if not Assigned(AParent) then
    raise Exception.Create('You must assign a parent.');
  FSoundCorrect := SDLAudio.SoundFromName('acerto');
  FSoundWrong   := SDLAudio.SoundFromName('erro');
  FSoundCorrect.SetOnStop(@ConsequenceDone);
  FSoundWrong.SetOnStop(@ConsequenceDone);

  with TrialKeys do begin
    FHasConsequence := AParameters.Values[HasConsequence].ToBoolean;
  end;
  with MTSKeys do begin
    LRelation := AParameters.Values[Relation];
    SampleLetter := ExtractDelimited(1,LRelation,['-']);
    ComparLetter := ExtractDelimited(2,LRelation,['-']);
    LSamples := AParameters.Values[Samples].ToInteger;
    LComparisons := AParameters.Values[Comparisons].ToInteger;

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

    if (FMTSModality.Comparisons = ModalityA) or
       (FMTSModality.Comparisons = ModalityD)
    then begin
      FButton := TButton.Create(Self);
      FButton.LoadFromFile(Pool.AssetsBasePath+'ConfirmButton'+IMG_EXT);
      FButton.Parent := TCustomRenderer(AParent);
      FButton.OnClick:=@ButtonClick;
    end;

    with SDLAudio.RecorderDevice do begin
      if FMTSModality.Comparisons = ModalityD then begin
        LComparisons := 1;
        Open;
      end else begin
        Close;
      end;
    end;
  end;

  NewGridItems(LSamples, LComparisons, goTopToBottom);
end;

procedure TMTSStimuli.Start;
var
  LStimulus : IStimulus;
begin
  for LStimulus in FSamples do
    LStimulus.Start;
end;

procedure TMTSStimuli.Stop;
var
  LStimulus : IStimulus;
begin
  for LStimulus in FComparisons do
    LStimulus.Stop;

  for LStimulus in FSamples do
    LStimulus.Stop;
end;

end.

