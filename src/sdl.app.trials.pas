{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.trials;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, Generics.Collections
  , SDL2
  , sdl.timer
  , sdl.app.controls.custom
  , sdl.app.trials.types
  , sdl.app.navigator.contract
  , sdl.app.navigable.contract
  , sdl.app.trials.contract
  , sdl.app.stimuli.contract
  , sdl.app.stimuli
  , sdl.app.events.abstract
  , sdl.app.events.custom
  , sdl.app.graphics.text
  , session.configuration
  , eye.tracker.types
  ;

type

  TStimuliList = specialize TObjectList<TStimuli>;
  TIStimuliList = specialize TList<IStimuli>;

  { TTrial }

  TTrial = class(TSDLControl, ITrial, INavigable)
    private
      FName: string;
      FNavigator: ITableNavigator;
      FText : TText;
      FParent : TSDLControl;
      FLimitedHoldTimer : TSDLTimer;
      FTestMode: Boolean;
      FVisible: Boolean;
      FIStimuli : IStimuli;
      FIStimuliList : TIStimuliList;
      FInterTrialInterval : Cardinal;
      FConsequenceInterval : Cardinal;
      FHasInstructions : Boolean;
      FHasCalibration  : Boolean;
      procedure SetNavigator(AValue: ITableNavigator);
      procedure SetParent(AValue: TSDLControl);
      procedure SetTestMode(AValue: Boolean);
      procedure EndStarterCallBack(Sender : TObject);
      procedure CreateStartersIfRequired;
      procedure GazeOnScreen(AGazes : TNormalizedGazes);
      procedure Pass(AGazes : TNormalizedGazes);
    protected
      FHasConsequence : Boolean;
      FResult : TTrialResult;
      FStimuliList : TStimuliList;
      FOnTrialEnd : TNotifyEvent;
      FConfiguration : TTrialConfiguration;
      procedure Paint; override;
      procedure EndTrialCallBack(Sender : TObject);
      procedure MouseMove(Sender:TObject; Shift: TCustomShiftState; X, Y: Integer); override;
      procedure MouseDown(Sender:TObject; Shift: TCustomShiftState; X, Y: Integer); override;
      procedure MouseUp(Sender:TObject; Shift: TCustomShiftState; X, Y: Integer); override;
      procedure SetOnTrialEnd(ANotifyEvent: TNotifyEvent);
      procedure SetTrialConfiguration(ATrialData: TTrialConfiguration); virtual;
      function GetOnTrialEnd: TNotifyEvent;
      function GetTrialConfiguration: TTrialConfiguration;
      function GetIStimuli : IStimuli; virtual; abstract;
      function MyResult : TTrialResult; virtual;
      //function Header : string; virtual; abstract;
      function ToData : string; virtual;
      procedure UpdateNavigator;
    public
      constructor Create; override;
      destructor Destroy; override;
      function ConsequenceDelay: Cardinal; virtual;
      function ConsequenceInterval: Cardinal; virtual;
      function InterTrialInterval : Cardinal; virtual;
      function AsITrial : ITrial;
      procedure EndTrial; virtual;
      procedure Show; virtual;
      procedure Hide; virtual;
      procedure UpdateController;
      procedure DoExpectedResponse;
      property Visible : Boolean read FVisible;
      property Data : TTrialConfiguration read GetTrialConfiguration write SetTrialConfiguration;
      property OnTrialEnd : TNotifyEvent read GetOnTrialEnd write SetOnTrialEnd;
      property TestMode : Boolean read FTestMode write SetTestMode;
      property Parent : TSDLControl read FParent write SetParent;
      property Name : string read FName write FName;
      property Navigator : ITableNavigator read FNavigator write SetNavigator;
  end;

const
  SESSION_TRIALEND = SDL_USEREVENT+0;

implementation

uses
  Pupil.Queue
  , eye.tracker.client
  , sdl.app.renderer.validation
  , sdl.app.video.methods
  , sdl.app.stimuli.instruction
  , sdl.app.stimuli.calibration
  , sdl.app.paintable.contract
  , sdl.app.clickable.contract
  , sdl.app.moveable.contract
  , sdl.app.lookable.contract
  , session.constants.trials
  , session.loggers.writerow.timestamp
  , sdl.app.mouse;

{ TTrial }

constructor TTrial.Create;
begin
  inherited Create;
  FRect := MonitorFromWindow;
  if Assigned(SDLEvents) then begin
    SDLEvents.AssignEvents;
    SDLEvents.OnMouseButtonDown := AsIClickable.GetSDLMouseButtonDown;
    SDLEvents.OnMouseButtonUp := AsIClickable.GetSDLMouseButtonUp;
    SDLEvents.OnMouseMotion := AsIMoveable.GetSDLMouseMotion;
    SDLEvents.OnGazeOnScreen := @GazeOnScreen;
  end;

  FVisible := False;
  FTestMode := False;
  FStimuliList := TStimuliList.Create;
  FIStimuliList := TIStimuliList.Create;
  FText := TText.Create;
  FLimitedHoldTimer := TSDLTimer.Create;
  FNavigator := nil;
  FLimitedHoldTimer.Interval := 0;
  FInterTrialInterval := 0;
  FConsequenceInterval := 0;
  FHasConsequence := True;
  FHasInstructions := False;
  FHasCalibration := False;

  with TrialKeys do begin
    RegisterParameter(LimitedHoldKey,
      @FLimitedHoldTimer.Interval, FLimitedHoldTimer.Interval);
    RegisterParameter(InterTrialIntervalKey,
      @FInterTrialInterval, FInterTrialInterval);
    RegisterParameter(ConsequenceIntervalKey,
      @FConsequenceInterval, FConsequenceInterval);
    RegisterParameter(HasConsequenceKey,
      @FHasConsequence, FHasConsequence);
    RegisterParameter(HasInstructionKey,
      @FHasInstructions, FHasInstructions);
    RegisterParameter(HasCalibrationKey,
      @FHasCalibration, FHasCalibration);
  end;
end;

destructor TTrial.Destroy;
begin
  if Assigned(SDLEvents) then begin
    SDLEvents.OnMouseButtonDown := nil;
    SDLEvents.OnMouseButtonUp := nil;
    SDLEvents.OnMouseMotion := nil;
    SDLEvents.OnUserEvent:=nil;
    SDLEvents.OnGazeOnScreen :=@Pass;
  end;
  FConfiguration.Parameters := nil;

  FLimitedHoldTimer.Free;
  FStimuliList.Free;
  FIStimuliList.Free;
  FText.Free;
  inherited Destroy;
end;

function TTrial.ConsequenceDelay: Cardinal;
begin
  Result := 0;
end;

function TTrial.ConsequenceInterval: Cardinal;
begin
  if FTestMode or
     (FResult = Hit) then begin
    Result := 0;
  end else begin
    if FHasConsequence then begin
      Result := FConsequenceInterval;
    end;
  end;
end;

function TTrial.InterTrialInterval: Cardinal;
begin
  if FTestMode or
     ((FResult = Miss) and FHasConsequence) then begin
    Result := 0;
  end else begin
    Result := FInterTrialInterval;
  end;
end;

function TTrial.AsITrial: ITrial;
begin
  Result := Self as ITrial;
end;

procedure TTrial.MouseMove(Sender: TObject; Shift: TCustomShiftState; X,
  Y: Integer);
var
  Child : TObject;
  SDLPoint : TSDL_Point;
  IChild : IMoveable;
begin
  if FVisible then begin
    for Child in FChildren do begin
      SDLPoint.x := X;
      SDLPoint.y := Y;
      IChild := IMoveable(TSDLControl(Child));
      if IChild.PointInside(SDLPoint) then begin
        if not IChild.MouseInside then begin
          IChild.MouseInside:=True;
          IChild.MouseEnter(Sender);
          GPaintingInvalidated := True;
        end;
        IChild.MouseMove(Sender, Shift, X, Y);
      end else begin
        if IChild.MouseInside then begin
          IChild.MouseInside:=False;
          IChild.MouseExit(Sender);
          GPaintingInvalidated := True;
        end;
      end;
    end;
  end;
end;

procedure TTrial.MouseDown(Sender:TObject; Shift: TCustomShiftState; X, Y: Integer);
var
  Child : TObject;
  SDLPoint : TSDL_Point;
  IChild : IClickable;
begin
  if FVisible then begin
    for Child in FChildren do begin
      SDLPoint.x := X;
      SDLPoint.y := Y;
      IChild := IClickable(TSDLControl(Child));
      if IChild.PointInside(SDLPoint) then
        IChild.MouseDown(Sender, Shift, X, Y);
    end;
  end;
end;

procedure TTrial.MouseUp(Sender: TObject; Shift: TCustomShiftState; X,
  Y: Integer);
var
  Child : TObject;
  SDLPoint : TSDL_Point;
  IChild   : IClickable;
begin
  if FVisible then begin
    for Child in FChildren do begin
      SDLPoint.x := X;
      SDLPoint.y := Y;
      IChild := IClickable(TSDLControl(Child));
      if IChild.PointInside(SDLPoint) then
        IChild.MouseUp(Sender, Shift, X, Y);
    end;
  end;
end;

procedure TTrial.EndTrial;
var
  event : TSDL_Event;
begin
  Hide;
  event.type_ := SESSION_TRIALEND;
  event.user.data1 := Pointer(Self);
  SDL_PushEvent(@event);
end;

procedure TTrial.EndTrialCallBack(Sender: TObject);
var
  LStimuli : IStimuli;
begin
  if Sender is IStimuli then begin
    LStimuli := Sender as IStimuli;
    if GetIStimuli.CustomName = LStimuli.CustomName then begin
      FResult := LStimuli.MyResult;
      EndTrial;
    end;
  end;

  if Sender is TSDLTimer then begin
    FIStimuli.Stop;
    FResult := TTrialResult.None;
    EndTrial;
  end;
end;

procedure TTrial.EndStarterCallBack(Sender: TObject);
begin
  with FIStimuliList do begin
    if Count > 0 then begin
      FIStimuli.Stop;
      FIStimuli := Extract(Last);
      if Sender = Self then begin
        { do nothing }
      end else begin
        Show;
      end;
    end;
  end;
end;

// todo: refactor starters as an IStimuliList to allow many instructions ...
procedure TTrial.CreateStartersIfRequired;
var
  LInstruction : TInstructionStimuli;
  LCalibration : TCalibrationStimuli;
  LIStimuli : IStimuli;
begin
  FIStimuliList.Add(FIStimuli);

  if FHasInstructions then begin
    LInstruction := TInstructionStimuli.Create;
    LInstruction.OnFinalize := @EndStarterCallBack;
    FStimuliList.Add(LInstruction);

    LIStimuli := LInstruction as IStimuli;
    LIStimuli.Load(FConfiguration.Parameters, Self);
    FIStimuliList.Add(LIStimuli);
  end;

  if TEyeTrackerClient.Exists and FHasCalibration then begin
    LCalibration := TCalibrationStimuli.Create;
    LCalibration.OnFinalize := @EndStarterCallBack;
    FStimuliList.Add(LCalibration);

    LIStimuli := LCalibration as IStimuli;
    LIStimuli.Load(FConfiguration.Parameters, Self);
    FIStimuliList.Add(LIStimuli);
  end;
  EndStarterCallBack(Self);
end;

procedure TTrial.GazeOnScreen(AGazes : TNormalizedGazes);
var
  Child : TObject;
  Gaze : TSDL_Point;
  IChild : ILookable;
  i: Integer;
begin
  if FVisible then begin
    for i := Low(AGazes) to High(AGazes) do begin
      for Child in FChildren do begin
        Gaze := NormToScreen(AGazes[i], FRect);

        IChild := ILookable(TSDLControl(Child));
        if IChild.IsGazeInside(Gaze, 50) then begin
          if not IChild.GazeInside then begin
            IChild.GazeInside:=True;
            IChild.GazeEnter(Self);
            GPaintingInvalidated := True;
            if FNavigator <> nil then begin
              FNavigator.SelectTarget(TSDLControl(Child).AsISelectable);
            end;
          end;
          IChild.GazeMove(Self, GetShiftState, Gaze.X, Gaze.Y);
        end else begin
          if IChild.GazeInside then begin
            IChild.GazeInside:=False;
            IChild.GazeExit(Self);
            GPaintingInvalidated := True;
          end;
        end;
      end;
    end;
  end;
end;

procedure TTrial.Pass(AGazes: TNormalizedGazes);
begin
  { do nothing }
end;


procedure TTrial.SetParent(AValue: TSDLControl);
begin
  if FParent = AValue then Exit;
  FParent := AValue;
end;

procedure TTrial.UpdateNavigator;
var
  LNaviable : INavigable;
begin
  LNaviable := FIStimuli.AsINavigable;
  if LNaviable <> nil then begin
    LNaviable.SetNavigator(Navigator);
  end;
end;

procedure TTrial.UpdateController;
var
  LNaviable : INavigable;
begin
  LNaviable := FIStimuli.AsINavigable;
  if LNaviable <> nil then begin
    LNaviable.SetNavigator(Navigator);
    LNaviable.UpdateNavigator;
  end;
end;

procedure TTrial.SetNavigator(AValue: ITableNavigator);
begin
  if FNavigator = AValue then Exit;
  FNavigator := AValue;
  FNavigator.SetBaseControl(AsISelectable);
end;

procedure TTrial.SetTestMode(AValue: Boolean);
begin
  if FTestMode = AValue then Exit;
  FTestMode := AValue;
  FText.FontName := 'Raleway-Regular';
  FText.FontSize := 50;
  FText.Load(Name);
  FText.Parent := Self;
  FText.Centralize;
end;

procedure TTrial.Paint;
var
  Child : TObject;
begin
  if FVisible then begin
    for Child in FChildren do begin
      IPaintable(TSDLControl(Child)).Paint;
    end;
  end;
end;

procedure TTrial.SetOnTrialEnd(ANotifyEvent: TNotifyEvent);
begin
  if FOnTrialEnd = ANotifyEvent then Exit;
  FOnTrialEnd := ANotifyEvent;
end;

procedure TTrial.SetTrialConfiguration(ATrialData: TTrialConfiguration);
begin
  FConfiguration := ATrialData;
  FIStimuli := GetIStimuli;
  FIStimuli.Load(FConfiguration.Parameters, Self);
  if Assigned(FConfiguration.Parameters) then begin
    LoadParameters(FConfiguration.Parameters);
    CreateStartersIfRequired;
  end;
end;

function TTrial.GetOnTrialEnd: TNotifyEvent;
begin
  Result := FOnTrialEnd;
end;

function TTrial.GetTrialConfiguration: TTrialConfiguration;
begin
  Result := FConfiguration;
end;

function TTrial.MyResult: TTrialResult;
begin
  Result := FResult;
end;

function TTrial.ToData: string;
begin
  Result := GetIStimuli.ToData;
end;

procedure TTrial.Show;
var
  Point : TSDL_Point = (x:0; y:0);
begin
  Mouse.MoveTo(Point);
  if TestMode then begin
    DoExpectedResponse;
  end else begin
    UpdateNavigator;
    FIStimuli.Start;

    if    (not FIStimuli.IsStarter)
      and (FLimitedHoldTimer.Interval > 0) then begin
      FLimitedHoldTimer.OnTimer := @EndTrialCallBack;
      FLimitedHoldTimer.Start;
    end;

    FVisible := True;

    with FIStimuli do begin
      Timestamp(CustomName+'.Show', Selectables.ToJSON);
    end;
  end;
  GPaintingInvalidated := True;
end;

procedure TTrial.Hide;
begin
  if TestMode then begin
    // FVisible := False;
  end else begin
    FVisible := False;
    Timestamp(FIStimuli.CustomName+'.Hide');
    FIStimuli.Stop;
    FLimitedHoldTimer.Stop;
  end;
  GPaintingInvalidated := True;
end;

// test mode
procedure TTrial.DoExpectedResponse;
begin
  if TestMode then begin
    FVisible := True;
    FText.Show;
  end else begin

  end;
  FIStimuli.DoExpectedResponse;
end;

end.

