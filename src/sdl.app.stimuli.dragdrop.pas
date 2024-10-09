{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço, Universidade Federal do Pará.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.stimuli.dragdrop;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Generics.Collections
  , sdl.app.stimuli.contract
  , sdl.app.stimuli
  , sdl.app.events.abstract
  , sdl.app.graphics.animation
  , sdl.app.graphics.picture.dragdrop
  , sdl.app.grids.types
  , sdl.app.grids
  , sdl.app.audio.contract
  , sdl.app.trials.types
  , sdl.app.renderer.thread
  ;

type

  TDragDropablePictures = specialize TList<TDragDropablePicture>;
  TAnimations = specialize TList<TAnimation>;

  { TDragDropStimuli }

  TDragDropStimuli = class(TStimuli)
  private
    FAutoAnimateOnStart : Boolean;
    FWrongDragDrops : integer;
    FRenderer : TRendererThread;
    FResult : TTrialResult;
    FSoundRight : ISound;
    FSoundWrong : ISound;
    FOnDragDropDone: TNotifyEvent;
    FOnOtherDragDrop: TDragDropEvent;
    FOnRightDragDrop: TDragDropEvent;
    FOnWrongDragDrop: TDragDropEvent;
    FComparisons : TDragDropablePictures;
    FSamples : TDragDropablePictures;
    FAnimation : TAnimation;
    FDoneAnimations : TAnimations;
    FGridOrientation : TGridOrientation;
    function GetRandomSample : TDragDropablePicture;
    function ToJSON(ADrag, ADrop : TDragDropablePicture) : string; overload;
    function ToJSON(ASamples, AComparisons: TDragDropablePictures) : string; overload;
    procedure OtherDragDrop(Sender, Source: TObject; X, Y: Integer);
    procedure RightDragDrop(Sender, Source: TObject; X, Y: Integer);
    procedure SetFocus(Sender:TObject; Shift: TCustomShiftState; X, Y: Integer);
    procedure SetOnDragDropDone(AValue: TNotifyEvent);
    procedure SetOnOtherDragDrop(AValue: TDragDropEvent);
    procedure SetOnRightDragDrop(AValue: TDragDropEvent);
    procedure SetOnWrongDragDrop(AValue: TDragDropEvent);
    procedure FreeGridItems;
    procedure WrongDragDrop(Sender, Source: TObject; X, Y: Integer);
    procedure Animate(ASample : TDragDropablePicture);
    procedure SoundStart(Sender: TObject);
    procedure SoundStop(Sender: TObject);
  protected
    function MyResult : TTrialResult; override;
    function Header : string; override;
    function ToData : string; override;
  public
    constructor Create; override;
    destructor Destroy; override;
    //procedure ResetGrid;
    function AsIStimuli : IStimuli;
    procedure Reboot;
    procedure DoExpectedResponse; override;
    procedure Load(AParameters : TStringList; AParent: TObject); override;
    procedure Start; override;
    procedure Stop; override;
    property OnDragDropDone : TNotifyEvent read FOnDragDropDone write SetOnDragDropDone;
    property OnRightDragDrop : TDragDropEvent read FOnRightDragDrop write SetOnRightDragDrop;
    property OnWrongDragDrop : TDragDropEvent read FOnWrongDragDrop write SetOnWrongDragDrop;
    property OnOtherDragDrop : TDragDropEvent read FOnOtherDragDrop write SetOnOtherDragDrop;
  end;

implementation

uses
  Math
  , StrUtils
  , Devices.RS232i
  , sdl.app.stimuli.dragdrop.types
  , sdl.colors
  , sdl.app.controls.custom
  , sdl.app.audio
  , session.constants.trials.dragdrop
  , session.constants.mts
  , session.pool
  , session.strutils
  , session.loggers.writerow
  , session.loggers.writerow.timestamp
  ;

{ TDragDropStimuli }

procedure TDragDropStimuli.DoExpectedResponse;
begin

end;

procedure TDragDropStimuli.Load(AParameters: TStringList; AParent: TObject);
type
  TParametersDictionary = specialize TDictionary<string, string>;
var
  S1           : string;
  LSampleKey   : string;
  LComparKey   : string;
  SampleLetter : string;
  ComparLetter : string;
  SomeKeyIsMissing : Boolean;
  GridSize      : Byte;
  LSamples      : integer;
  LComparisons  : integer;
  LItem : TDragDropablePicture;
  i: Integer;

  Name : string;
  Value : string;
  Parameters : TParametersDictionary;

  function DragDropToGridOrientation(
      ADragDropOrientation : TDragDropOrientation) : TGridOrientation;
  begin
    case ADragDropOrientation of
      None : Result := TGridOrientation.goNone;
      TopToBottom : Result := TGridOrientation.goTopToBottom;
      BottomToTop : Result := TGridOrientation.goBottomToTop;
      LeftToRight : Result := TGridOrientation.goLeftToRight;
      RightToLeft : Result := TGridOrientation.goRightToLeft;
      Random : Result:= Grid.GetRandomGridOrientation;
    end;
  end;

  procedure NewGridItems(ASamples, AComparisons: integer;
    AGridOrientation: TGridOrientation);
  var
    LItem : TDragDropablePicture;
    LComparison : TDragDropablePicture;
    LComparisons : TDragDropablePictures;
    i : integer;
  begin
    Grid.UpdatePositions(GridSize,
      ASamples, AComparisons, AGridOrientation,
      Grid.FixedSample, Grid.FixedComparison);
    with Grid.RandomPositions do begin
      LComparisons := TDragDropablePictures.Create;
      for i := low(Comparisons) to high(Comparisons) do
      begin
        LItem := TDragDropablePicture.Create;
        LItem.BoundsRect := Comparisons[i].Rect;
        LItem.SetOriginalBounds;
        LItem.Position := Comparisons[i].Position;
        LItem.IsSample := False;
        Comparisons[i].Item := LItem as TObject;
        FComparisons.Add(LItem);
        LComparisons.Add(LItem);
      end;

      for i := low(Samples) to high(Samples) do
      begin
        LItem := TDragDropablePicture.Create;
        LItem.OnMouseDown := @SetFocus;
        LItem.OnRightDragDrop:=@RightDragDrop;
        LItem.OnWrongDragDrop:=@WrongDragDrop;
        LItem.OnOtherDragDrop:=@OtherDragDrop;
        LItem.BoundsRect := Samples[i].Rect;
        LItem.SetOriginalBounds;
        LItem.Draggable := True;
        LItem.IsSample := True;
        case i of
          0 : // do nothing;

          else begin                     // making sure that we have always
            LComparisons.Exchange(0, i); // the right comparison as the first one
          end;                           // inside the sample targets
        end;
        for LComparison in LComparisons do
          LItem.AddOrderedChoice(LComparison);

        LItem.Position := Samples[i].Position;
        Samples[i].Item := LItem as TObject;
        FSamples.Add(LItem);
      end;
      LComparisons.Free;
    end;
    AppendToTrialHeader(Pool.Session.Trial.Events.Header);
    AppendToTrialHeader(Grid.Header);
    AppendToTrialHeader(Header);
  end;

begin
  inherited Load(AParameters, AParent);
  Grid.FixedSample := False;
  Grid.FixedComparison := False;

  FAnimation.Parent := TSDLControl(AParent);

  FSoundRight := SDLAudio.SoundFromName('acerto');
  FSoundRight.SetOnStart(@SoundStart);
  FSoundRight.SetOnStop(@SoundStop);

  FSoundWrong := SDLAudio.SoundFromName('erro');
  FSoundWrong.SetOnStart(@SoundStart);
  FSoundWrong.SetOnStop(@SoundStop);

  Parameters := TParametersDictionary.Create;
  try
    for i := 0 to AParameters.Count-1 do begin
      AParameters.GetNameValue(i, Name, Value);
      Parameters.Add(Name, Value);
    end;

    //Cursor := StrToIntDef(Parameters['Cursor'], -1);

    with DragDropKeys, MTSKeys do begin
      //ChannelDragMouseMoveFactor :=
        //Parameters[DragMoveFactor].ToInteger;
      //DragMouseMoveMode :=
      //  Parameters[SamplesDragMode].ToDragMouseMoveMode;

      case Parameters[GridSizeKey].ToInteger of
        1..255: GridSize := Byte(Parameters[GridSizeKey].ToInteger);
        otherwise
          raise Exception.Create('GridSize not supported.');
      end;

      FGridOrientation := DragDropToGridOrientation(
        Parameters[DragDropOrientationKey].ToDragDropOrientation);
      FAutoAnimateOnStart := Parameters[AutoAnimateOnStartKey].ToBoolean;

      LSamples := Parameters[SamplesKey].ToInteger;
      LComparisons := Parameters[ComparisonsKey].ToInteger;

      SomeKeyIsMissing := False;
      for i := 0 to LSamples - 1 do begin
        LSampleKey := SampleKey+(i+1).ToString;
        SomeKeyIsMissing := not Parameters.ContainsKey(LSampleKey);
        if SomeKeyIsMissing then begin
          raise Exception.Create('TDragDropStimuli.Load: Missing Sn key.');
          Break;
        end;
      end;

      if not SomeKeyisMissing then begin
        for i := 0 to LComparisons - 1 do begin
          LComparKey := ComparisonKey+(i+1).ToString;
          SomeKeyIsMissing := not Parameters.ContainsKey(LComparKey);
          if SomeKeyIsMissing then begin
            raise Exception.Create('TDragDropStimuli.Load: Missing Cn key.');
            Break;
          end;
        end;
      end;

      if (LSamples*LComparisons) < (GridSize*2) then begin
        raise Exception.Create('Gridsize smaller than samples+comparisons');
        Exit;
      end;

      // instatiate grid objects
      NewGridItems(LSamples, LComparisons, FGridOrientation);

      // populate grid objects
      // if no Sn, Cn key is missing, then use them to load media files
      // else use Relation parameter
      if not SomeKeyisMissing then begin
        with Grid.RandomPositions do begin
          for i := Low(Comparisons) to High(Comparisons) do begin
            LItem := Comparisons[i].Item as TDragDropablePicture;
            //LItem.Cursor := Cursor;
            LComparKey := ComparisonKey+(i+1).ToString;
            LItem.LoadFromFile(AsImage(Parameters[LComparKey]));
            LItem.Parent := TSDLControl(AParent);
          end;

          for i := Low(Samples) to High(Samples) do begin
            LItem := Samples[i].Item as TDragDropablePicture;
            //LItem.Cursor := Cursor;
            LSampleKey := SampleKey+(i+1).ToString;
            LItem.LoadFromFile(AsImage(Parameters[LSampleKey]));
            LItem.UpdateDistance;
            LItem.Parent := TSDLControl(AParent);
            LItem.MoveToPoint(Parameters[DistanceKey].ToInteger);
          end;
        end;

      end else begin
        S1 := AParameters.Values[RelationKey];
        SampleLetter := ExtractDelimited(1,S1,['-']);
        ComparLetter := ExtractDelimited(2,S1,['-']);

        with Grid.RandomPositions do begin
          for i := Low(Comparisons) to High(Comparisons) do begin
            LItem := Comparisons[i].Item as TDragDropablePicture;
            //LItem.Cursor := Cursor;
            LItem.LoadFromFile(AsImage(ComparLetter+(i+1).ToString));
            LItem.Parent := TSDLControl(AParent);
          end;

          for i := Low(Samples) to High(Samples) do begin
            LItem := Samples[i].Item as TDragDropablePicture;
            //LItem.Cursor := Cursor;
            LItem.LoadFromFile(AsImage(SampleLetter+(i+1).ToString));
            LItem.UpdateDistance;
            LItem.Parent := TSDLControl(AParent);
            LItem.MoveToPoint(Parameters[DistanceKey].ToInteger);
          end;
        end;
      end;
    end;
  finally
    Parameters.Free;
  end;
end;

procedure TDragDropStimuli.Start;
var
  LItem : TDragDropablePicture;
begin
  for LItem in FComparisons do LItem.Show;
  for LItem in FSamples do LItem.Show;

  if FAutoAnimateOnStart then begin
    LItem := GetRandomSample;
    Animate(LItem);
  end;
  FRenderer.Render;
end;

procedure TDragDropStimuli.Stop;
var
  LItem : TDragDropablePicture;
  LAnimation: TAnimation;
begin
  FRenderer.Stop;
  for LItem in FComparisons do
    LItem.Hide;
  for LItem in FSamples do
    LItem.Hide;

  FAnimation.Stop;
  FAnimation.Hide;
  for LAnimation in FDoneAnimations do
    LAnimation.Hide;
end;

procedure TDragDropStimuli.OtherDragDrop(Sender, Source: TObject;
  X, Y: Integer);
var
  LSample, LSourceSample: TDragDropablePicture;
begin
  LSourceSample := Source as TDragDropablePicture;
  for LSample in FSamples do begin
    if LSample = LSourceSample then Continue;
    if LSample.IntersectsWith(LSourceSample) then begin
      LSourceSample.ToOriginalBounds;
      Timestamp(ClassName+'.OtherDragDrop', ToJSON(LSourceSample, nil));
    end;
  end;
  //LItem.Color := clWhite;


  if Assigned(OnOtherDragDrop) then
    OnOtherDragDrop(Sender, Source, X, Y);
end;

procedure TDragDropStimuli.RightDragDrop(Sender, Source: TObject;
  X, Y: Integer);
var
  Sample : TDragDropablePicture;
  Comparison : TDragDropablePicture;
  LAnimation : TAnimation;
  FDragDropDone : Boolean = False;
  S1 : string;
begin
  FSoundRight.Play;
  if Assigned(RS232) then begin
    RS232.Dispenser;
  end;

  Sample := Source as TDragDropablePicture;
  Comparison := Sender as TDragDropablePicture;

  Timestamp(ClassName+'.RightDragDrop', ToJSON(Sample, Comparison));

  //Sample.Color := clGreen;
  case FGridOrientation of
    TGridOrientation.goTopToBottom : begin
      Sample.Left := Comparison.Left;
      Sample.Top := Comparison.Top - Sample.Height - 10;
    end;

    TGridOrientation.goBottomToTop : begin
      Sample.Left := Comparison.Left;
      Sample.Top := Comparison.Top + Sample.Height + 10;
    end;

    TGridOrientation.goLeftToRight : begin
      Sample.Left := Comparison.Left - Sample.Width - 10;
      Sample.Top := Comparison.Top;
    end;

    TGridOrientation.goRightToLeft : begin
      Sample.Left := Comparison.Left + Sample.Width + 10;
      Sample.Top := Comparison.Top;
    end;

    TGridOrientation.goNone : begin
      Sample.Hide;
      Comparison.Hide;
    end;

    otherwise begin
      WriteStr(S1, FGridOrientation);
      raise Exception.Create(
        'TDragDropStimuli.RightDragDrop: ' + S1);
    end;
  end;

  LAnimation := TAnimation.Create;
  // LAnimation.Parent := FAnimation.Parent;
  // LAnimation.Cursor:=Cursor;
  LAnimation.Join(Sample, Comparison, FGridOrientation);
  LAnimation.Show;
  FDoneAnimations.Add(LAnimation);

  Sample.Draggable:=False;

  for Sample in FSamples do
    if Sample.Draggable then begin
      FDragDropDone := False;
      Animate(Sample);
      Break;
    end else begin
      //Sample.EdgeColor:=clInactiveCaption;
      FDragDropDone := True;
    end;

  if Assigned(OnRightDragDrop) then
    OnRightDragDrop(Sender, Source, X, Y);

  if FDragDropDone then begin
    //FResult := Hit; todo: dragdrop trial have different hit types
    Pool.Counters.Hit;
    FResult := Hit;
    FAnimation.Stop;
    FAnimation.Hide;
    Timestamp(ClassName+'.DragDropDone');
    Finalize;
  end;
end;

procedure TDragDropStimuli.SetFocus(Sender:TObject; Shift: TCustomShiftState; X, Y: Integer);
var
  LSample : TDragDropablePicture;
begin
  LSample := Sender as TDragDropablePicture;
  if Assigned(OnResponse) then begin
    OnResponse(LSample);
  end;

  if LSample.Draggable then begin
    if (FAnimation.Sibling as TDragDropablePicture) <> LSample then begin
      if Assigned(FAnimation.Sibling) then begin
        FAnimation.Sibling.EdgeColor:=clGray;
      end;
      Animate(LSample);
    end;
  end;
end;

procedure TDragDropStimuli.SetOnDragDropDone(AValue: TNotifyEvent);
begin
  if FOnDragDropDone=AValue then Exit;
  FOnDragDropDone:=AValue;
end;

procedure TDragDropStimuli.SetOnOtherDragDrop(AValue: TDragDropEvent);
begin
  if FOnOtherDragDrop=AValue then Exit;
  FOnOtherDragDrop:=AValue;
end;

procedure TDragDropStimuli.SetOnRightDragDrop(AValue: TDragDropEvent);
begin
  if FOnRightDragDrop=AValue then Exit;
  FOnRightDragDrop:=AValue;
end;

procedure TDragDropStimuli.SetOnWrongDragDrop(AValue: TDragDropEvent);
begin
  if FOnWrongDragDrop=AValue then Exit;
  FOnWrongDragDrop:=AValue;
end;

function TDragDropStimuli.GetRandomSample: TDragDropablePicture;
begin
  Result := FSamples[RandomRange(0, FSamples.Count)];
end;

function TDragDropStimuli.ToJSON(
  ADrag, ADrop: TDragDropablePicture): string;
var
  LDropString : string;
begin
  if ADrop = nil then begin
    LDropString := 'NA';
  end else begin
    LDropString := ADrop.ToJSON;
  end;

  Result := '{'+String.Join(',', [
    'drag:'+ ADrag.ToJSON,
    'drop:'+ LDropString])+'}';
end;

function TDragDropStimuli.ToJSON(
  ASamples, AComparisons: TDragDropablePictures): string;
var
  LSamples : array of string;
  LComparisons : array of string;
  i: Integer;
begin
  LSamples := Nil;
  LComparisons := Nil;

  SetLength(LSamples, ASamples.Count);
  for i := Low(LSamples) to High(LSamples) do begin
    LSamples[i] := ASamples[i].ToJSON;
  end;

  SetLength(LComparisons, AComparisons.Count);
  for i := Low(LComparisons) to High(LComparisons) do begin
    LSamples[i] := AComparisons[i].ToJSON;
  end;

  Result := '{'+
    String.Join(',',[
      'samples:'+'{'+ String.Join(',', LSamples) +'}',
      'comparisons:'+'{'+ String.Join(',', LComparisons) +'}'
    ])+'}';
end;

procedure TDragDropStimuli.FreeGridItems;
var
  i: Integer;
begin
  with Grid.RandomPositions do begin
    for i := low(Comparisons) to high(Comparisons) do begin
      Comparisons[i].Item.Free;
    end;

    for i := low(Samples) to high(Samples) do begin
      Samples[i].Item.Free
    end;
  end;
end;

procedure TDragDropStimuli.WrongDragDrop(Sender, Source: TObject;
  X, Y: Integer);
var
  Comparison, Sample : TDragDropablePicture;
begin
  Inc(FWrongDragDrops);
  FSoundWrong.Play;
  Comparison := Sender as TDragDropablePicture;
  Sample := Source as TDragDropablePicture;
  Timestamp('WrongDragDrop', ToJSON(Sample, Comparison));

  Sample.ToOriginalBounds;
  //FAnimation.Animate(Sample);
  if Assigned(OnWrongDragDrop) then begin
    OnWrongDragDrop(Sender, Source, X, Y);
  end;
end;

procedure TDragDropStimuli.Animate(ASample: TDragDropablePicture);
begin
  //FAnimation.Cursor := Cursor;
  //ASample.UpdateDragMouseMoveMode;
  FAnimation.Animate(ASample);
  FAnimation.Show;
  FAnimation.BringToFront;
  ASample.BringToFront;
  //Parent.Invalidate;
end;

procedure TDragDropStimuli.SoundStart(Sender: TObject);
begin
  if (Sender as ISound) = FSoundRight then begin
    Timestamp('Hit.Start');
  end;

  if (Sender as ISound) = FSoundWrong then begin
    Timestamp('Miss.Start');
  end;
end;

procedure TDragDropStimuli.SoundStop(Sender: TObject);
begin
  if (Sender as ISound) = FSoundRight then begin
    Timestamp('Hit.Stop');
  end;

  if (Sender as ISound) = FSoundWrong then begin
    Timestamp('Miss.Stop');
  end;
end;

function TDragDropStimuli.MyResult: TTrialResult;
begin
  Result := FResult;
end;

function TDragDropStimuli.Header: string;
begin
  Result := 'WrongDragDrops';
end;

function TDragDropStimuli.ToData: string;
begin
  Result := FWrongDragDrops.ToString;
end;

constructor TDragDropStimuli.Create;
begin
  inherited Create;
  FAutoAnimateOnStart := False;
  FWrongDragDrops := 0;
  FSamples := TDragDropablePictures.Create;
  FComparisons := TDragDropablePictures.Create;
  FAnimation := TAnimation.Create;
  FDoneAnimations := TAnimations.Create;
  FRenderer := TRendererThread.Create;
end;

destructor TDragDropStimuli.Destroy;
var
  LDragDropablePicture : TDragDropablePicture;
  LAnimation : TAnimation;
begin
  FRenderer.Close;
  FRenderer.Terminate;
  FAnimation.Free;

  for LAnimation in FDoneAnimations do begin
    LAnimation.Free;
  end;
  FDoneAnimations.Free;

  for LDragDropablePicture in FSamples do begin
    LDragDropablePicture.Free;
  end;
  FSamples.Free;

  for LDragDropablePicture in FComparisons do begin
    LDragDropablePicture.Free;
  end;
  FComparisons.Free;
  inherited Destroy;
end;

function TDragDropStimuli.AsIStimuli: IStimuli;
begin
  Result := Self as IStimuli;
end;

procedure TDragDropStimuli.Reboot;
var
  i: Integer;
begin
  with Grid.RandomPositions do begin
    for i := Low(Samples) to High(Samples) do begin
      (Samples[i].Item as TDragDropablePicture).ToOriginalBounds;
    end;
  end;
  Timestamp('KeyPressed.Samples.ToOriginalBounds');
end;

end.
