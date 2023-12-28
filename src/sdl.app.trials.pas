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
  Classes, SysUtils, fgl
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

  TStimuliList = specialize TFPGObjectList<TStimuli>;

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
      FICalibration : IStimuli;
      FIInstruction : IStimuli;
      FInterTrialInterval : Cardinal;
      FConsequenceInterval : Cardinal;
      FHasInstructions : Boolean;
      FHasCalibration  : Boolean;
      procedure UpdateNavigator;
      procedure SetNavigator(AValue: ITableNavigator);
      procedure SetParent(AValue: TSDLControl);
      procedure SetTestMode(AValue: Boolean);
      procedure EndStarterCallBack(Sender : TObject);
      procedure CreateStartersIfRequired;
      procedure GazeOnScreen(Sender : TObject;  AGazes : TGazes);
    protected
      FHasConsequence : Boolean;
      FResult : TTrialResult;
      FStimuliList : TStimuliList;
      FOnTrialEnd : TNotifyEvent;
      FData : TTrialData;
      procedure Paint; override;
      procedure EndTrialCallBack(Sender : TObject);
      procedure MouseMove(Sender:TObject; Shift: TCustomShiftState; X, Y: Integer); override;
      procedure MouseDown(Sender:TObject; Shift: TCustomShiftState; X, Y: Integer); override;
      procedure MouseUp(Sender:TObject; Shift: TCustomShiftState; X, Y: Integer); override;
      procedure SetOnTrialEnd(ANotifyEvent: TNotifyEvent);
      procedure SetTrialData(ATrialData: TTrialData); virtual;
      function GetOnTrialEnd: TNotifyEvent;
      function GetTrialData: TTrialData;
      function GetIStimuli : IStimuli; virtual; abstract;
      function MyResult : TTrialResult; virtual;
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
      procedure DoExpectedResponse;
      property Visible : Boolean read FVisible;
      property Data : TTrialData read GetTrialData write SetTrialData;
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
    eye.tracker.client
  , sdl.app.video.methods
  , sdl.app.stimuli.instruction
  , sdl.app.stimuli.calibration.pupil
  , sdl.app.paintable.contract
  , sdl.app.clickable.contract
  , sdl.app.moveable.contract
  , sdl.app.lookable.contract
  , sdl.app.controller.manager
  , session.constants.trials
  , session.loggers.writerow.timestamp;

{ TTrial }

constructor TTrial.Create;
begin
  inherited Create;
  FRect := MonitorFromWindow;
  SDLEvents.AssignEvents;
  SDLEvents.OnMouseButtonDown := AsIClickable.GetSDLMouseButtonDown;
  SDLEvents.OnMouseButtonUp := AsIClickable.GetSDLMouseButtonUp;
  SDLEvents.OnMouseMotion := AsIMoveable.GetSDLMouseMotion;
  SDLEvents.OnGazeOnScreen := @GazeOnScreen;

  FICalibration := nil;
  FIInstruction := nil;
  FVisible := False;
  FTestMode := False;
  FStimuliList := TStimuliList.Create;
  FText := TText.Create;
  FLimitedHoldTimer := TSDLTimer.Create;

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
  SDLEvents.OnMouseButtonDown := nil;
  SDLEvents.OnMouseButtonUp := nil;
  SDLEvents.OnMouseMotion := nil;
  SDLEvents.OnUserEvent:=nil;
  FData.Parameters := nil;

  FLimitedHoldTimer.Free;
  FStimuliList.Free;
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
  if FTestMode then begin
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
        end;
        IChild.MouseMove(Sender, Shift, X, Y);
      end else begin
        if IChild.MouseInside then begin
          IChild.MouseInside:=False;
          IChild.MouseExit(Sender);
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

procedure DoEndTrial(PSelf: Pointer);
var
  event : TSDL_Event;
begin
  event.type_ := SESSION_TRIALEND;
  event.user.data1 := PSelf;
  SDL_PushEvent(@event);
end;

procedure TTrial.EndTrial;
begin
  Hide;
  DoEndTrial(Pointer(Self));
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
end;

// todo: refactor starters as an IStimuliList, call next stimuli here...
procedure TTrial.EndStarterCallBack(Sender: TObject);
begin
  if Sender is TInstructionStimuli then begin
    TInstructionStimuli(Sender).Stop;
    if Assigned(FICalibration) then begin
      FICalibration.Start;
    end else begin
      FIStimuli := GetIStimuli;
      Show;
    end;
  end;

  if Sender is TPupilCalibrationStimuli then begin
    TPupilCalibrationStimuli(Sender).Stop;
    FIStimuli := GetIStimuli;
    Show;
  end;

  //with FStimuliList do
  //  if Count > 0 then begin
  //    FIStimuli := Extract(Last);
  //  end;
end;

// todo: refactor starters as an IStimuliList to allow many instructions ...
procedure TTrial.CreateStartersIfRequired;
var
  LInstruction : TInstructionStimuli;
  LCalibration : TPupilCalibrationStimuli;
begin
  if FHasInstructions then begin
    LInstruction := TInstructionStimuli.Create;
    LInstruction.OnFinalize := @EndStarterCallBack;
    FIInstruction := LInstruction;
    FIInstruction.Load(FData.Parameters, Self);
    FIStimuli := FIInstruction;
    FStimuliList.Add(LInstruction);
  end;

  if TEyeTrackerClient.Exists and FHasCalibration then begin
    LCalibration := TPupilCalibrationStimuli.Create;
    LCalibration.OnFinalize := @EndStarterCallBack;
    FICalibration := LCalibration;
    FICalibration.Load(FData.Parameters, Self);
    FIStimuli := LCalibration;
    FStimuliList.Add(LCalibration);
  end;
end;

procedure TTrial.GazeOnScreen(Sender: TObject; AGazes: TGazes);
var
  Child : TObject;
  SDLPoint : TSDL_Point;
  IChild : ILookable;
  i: Integer;
begin
  if FVisible then begin
    if Length(AGazes) > 0 then begin
      for i := Low(AGazes) to High(AGazes) do begin
        for Child in FChildren do begin
          SDLPoint.x := AGazes[i].X;
          SDLPoint.y := AGazes[i].Y;
          IChild := ILookable(TSDLControl(Child));
          if IChild.PointInside(SDLPoint) then begin
            if not IChild.GazeInside then begin
              IChild.GazeInside:=True;
              IChild.GazeEnter(Sender);
            end;
            IChild.GazeMove(Sender, GetShiftState, AGazes[i].X, AGazes[i].Y);
          end else begin
            if IChild.GazeInside then begin
              IChild.GazeInside:=False;
              IChild.GazeExit(Sender);
            end;
          end;
        end;
      end;
    end;
  end;
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

procedure TTrial.SetTrialData(ATrialData: TTrialData);
begin
  FData := ATrialData;
  FIStimuli := GetIStimuli;
  FIStimuli.Load(FData.Parameters, Self);
  if Assigned(FData.Parameters) then begin
    LoadParameters(FData.Parameters);
    CreateStartersIfRequired;
  end;
end;

function TTrial.GetOnTrialEnd: TNotifyEvent;
begin
  Result := FOnTrialEnd;
end;

function TTrial.GetTrialData: TTrialData;
begin
  Result := FData;
end;

function TTrial.MyResult: TTrialResult;
begin
  Result := FResult;
end;

procedure TTrial.Show;
begin
  if TestMode then begin
    DoExpectedResponse;
  end else begin
    UpdateNavigator;
    FIStimuli.Start;
    if FLimitedHoldTimer.Interval > 0 then begin
      FLimitedHoldTimer.Start;
    end;
    FVisible := True;
    Timestamp(FIStimuli.CustomName+'.Show');
  end;
  SDL_ShowCursor(SDL_ENABLE);
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
end;

// test mode
procedure TTrial.DoExpectedResponse;
begin
  FVisible := True;
  FText.Show;
  FIStimuli.DoExpectedResponse;
end;

end.

